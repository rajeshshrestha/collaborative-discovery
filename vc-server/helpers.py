from random import randint
from tqdm import tqdm
import pprint
import os
import json
import subprocess as sp
import pandas as pd
import numpy as np
import sys
import pickle
from collections import Counter
from pprint import pprint

sys.path.insert(0, './charm/keywordSearch/charm')
import charm


def applyUserRepairs(d_dirty, s_in):
    d_rep = d_dirty
    s_df = pd.read_json(s_in, orient='index')
    print(s_df.index)
    cfd_metadata = pickle.load( open('./store/' + project_id + '/cfd_metadata.p', 'rb') )
    receiver = pickle.load( open('./store/' + project_id + '/receiver.p', 'rb') )
    current_iter = pickle.load( open('./store/' + project_id + '/current_iter.p', 'rb') )
    curr_iter_num = int('0x' + current_iter, 0)
    cfd_applied_map = pickle.load( open('./store/' + project_id))
    value_metadata = pickle.load( open('./store/' + project_id + '/value_metadata.p', 'rb') )

    changed_ids = list()
    num_times_applied_each_cfd = dict()
    for idx in s_df.index:
        for col in s_df.columns:
            if cfd_applied_map[-1][idx][col] in num_times_applied_each_cfd.keys():
                num_times_applied_each_cfd[cfd_applied_map[-1][idx][col]] += 1
            else:
                num_times_applied_each_cfd[cfd_applied_map[-1][idx][col]] = 1

    for idx in s_df.index:
        for col in s_df.columns:
            if d_rep.at[idx, col] != s_df.at[idx, col]:
                changed_ids.append(idx)
                try:
                    latest_match_idx = next(i for i in reversed(range(len(value_metadata[idx][col]['history']))) if value_metadata[idx][col]['history'][i][0] == s_df.at[idx, col])
                    charm.reinforce(receiver, value_metadata[idx][col]['history'][latest_match_index][1], latest_match_idx/(len(value_metadata[idx][col]['history']) - 1))
                    #charm.reinforce(receiver, value_metadata[idx][col]['history'][-1][1], -1/num_times_applied_each_cfd[value_metadata[idx][col]['history'][-1][1]])
                except ValueError:
                    last_cfd_id = value_metadata[idx][col]['history'][-1][1]
                    #charm.reinforce(receiver, last_cfd_id, -1/num_times_applied_each_cfd[last_cfd_id])
                d_rep.at[idx, col] = s_df.at[idx, col]
            else:
                last_cfd_id = value_metadata[idx][col]['history'][-1][1]
                charm.reinforce(receiver, last_cfd_id, 1/num_times_applied_each_cfd[last_cfd_id])



    #TODO: Find the CFD applied by the system for changes made by the user, and check if it was reverted to match a previous CFD


    #pickle.dump(receiver, open('./store/' + project_id + '/receiver.p', 'wb'))
    #pickle.dump(cfd_metadata, open('./store/' + project_id + '/cfd_metadata.p', 'wb'))

    return d_rep, changed_ids


def discoverCFDs(project_id, current_iter):
    #prev_iter = "{:08x}".format(int('0x'+current_iter, 0) - 1)
    dirty_fp = './store/' + project_id + '/00000001/data.csv'
    clean_fp = './store/' + project_id + '/' + current_iter + '/data.csv'

    process = sp.Popen(['./xplode-master/CTane', dirty_fp, clean_fp, '0.25', '2'], stdout=sp.PIPE, stderr=sp.PIPE, env={'LANG': 'C++'})
    output = process.communicate()[0].decode("utf-8")
    if process.returncode == 0:
        if output == '[NO CFDS FOUND]':
            return None
        else:
            output = np.array(output.split('\n'))[:-1]
            scores = output[:int(len(output)/2)]
            top_cfds = output[int(len(output)/2):]
            return np.array([{'cfd_id': i, 'cfd': top_cfds[i], 'score': scores[i]} for i in range(0, len(top_cfds))])
    else:
        return '[ERROR] CFD DISCOVERY FAILURE'


def addNewCfdsToList(top_cfds, project_id, current_iter, receiver=None):
    if os.path.isfile('./store/' + project_id + '/cfd_metadata.p'):
        cfd_metadata = pickle.load( open('./store/' + project_id + '/cfd_metadata.p', 'rb') )
        receiver = pickle.load( open('./store/' + project_id + '/receiver.p', 'rb') )
        for c in [tc for tc in top_cfds if tc['score'] > 0]:
            exists = False
            for idx, cfd in enumerate(cfd_metadata):
                lhs = c['cfd'].split(' => ')[0][1:-1]
                rhs = c['cfd'].split(' => ')[1]
                if cfd['lhs'] == lhs and cfd['rhs'] == rhs:
                    exists = True
                    cfd['num_found'] += 1
                    cfd['num_changes'][current_iter] = [0]
                    charm.reinforce(receiver, idx, c['score']/cfd['num_found'])
                    break
            if not exists:
                cfd_metadata.append(dict())
                cfd_metadata[-1]['lhs'] = c['cfd'].split(' => ')[0][1:-1]
                cfd_metadata[-1]['rhs'] = c['cfd'].split(' => ')[1]
                cfd_metadata[-1]['num_found'] = 1
                cfd_metadata[-1]['num_changes'] = dict()
                cfd_metadata[-1]['num_changes'][current_iter] = 0

                c['cfd_id'] = len(cfd_metadata) - 1
                charm.updateReceiver(receiver, [c])
                charm.reinforce(receiver, c['cfd_id'], c['score'])

        pickle.dump( receiver, open('./store/' + project_id + '/receiver.p', 'wb') )
        pickle.dump( cfd_metadata, open('./store/' + project_id + '/cfd_metadata.p', 'wb') )

    else:
        cfd_metadata = list()
        for c in [tc for tc in top_cfds if tc['score'] > 0]:
            cfd_metadata.append(dict())
            cfd_metadata[-1]['lhs'] = c['cfd'].split(' => ')[0][1:-1]
            cfd_metadata[-1]['rhs'] = c['cfd'].split(' => ')[1]
            cfd_metadata[-1]['num_found'] = 1
            cfd_metadata[-1]['num_changes'] = dict()
            cfd_metadata[-1]['num_changes'][current_iter] = 0

        scores = [c['score'] for c in top_cfds if c['score'] > 0]

        receiver = charm.prepareReceiver(project_id, top_cfds)
        for idx, _ in enumerate(cfd_metadata):
            charm.reinforce(receiver, idx, scores[idx])

        pickle.dump( receiver, open('./store/' + project_id + '/receiver.p', 'wb') )
        pickle.dump( cfd_metadata, open('./store/' + project_id + '/cfd_metadata.p', 'wb') )


def buildCover(d_rep, picked_cfds):
    cover = np.empty(len(d_rep.index), dtype=str)
    print(picked_cfds[0]['cfd'])
    print([c['cfd'] for c in picked_cfds])
    just_cfds = np.array([c['cfd'] for c in picked_cfds])
    for idx, row in d_rep.iterrows():
        relevant_cfds = []
        for cfd in just_cfds:
            lhs = cfd.split(' => ')[0][1:-1]
            rhs = cfd.split(' => ')[1]
            applies = True
            for lh in lhs.split(', '):
                if '=' in lh:
                    lh = np.array(lh.split('='))
                    if row[lh[0]] != lh[1]:
                        applies = False
                        break
            if applies:
                relevant_cfds.append(cfd)
        cover[idx] = '; '.join(relevant_cfds)

    d_rep['cover'] = cover
    return d_rep

# TODO; CHARM-INTEGRATED IMPLEMENTATION
def pickCfds(query, num_cfds):
    receiver = pickle.load( open('./store/' + project_id + '/receiver.p', 'rb') )
    rules, rule_id_list = charm.getRules(receiver, query, num_cfds)
    pickle.dump( open('./store/' + project_id + '/receiver.p', 'wb') )
    return rules, rule_id_list

# TODO; TEMPORARY IMPLEMENTATION
#def pickCfds(top_cfds, num_cfds):
#    just_cfds = np.array([c['cfd'] for c in top_cfds if float(c['score']) > 0])
#    just_scores = np.array([float(c['score']) for c in top_cfds if float(c['score']) > 0])
#    norm_scores = np.array([s/sum(just_scores) for s in just_scores])
#    if len(just_cfds) > 0:
#        picked_cfds = np.random.choice(just_cfds, num_cfds, p=norm_scores.astype('float64'))
#        return picked_cfds, picked_cfd_ids
#    else:
#        return None, None

def applyCfdList(project_id, d_rep, cfd_list, cfd_id_list, cfd_applied_map, current_iter):
    for idx in d_rep.index:
        cfd_applied_map[idx] = dict()
        for col in d_rep.columns:
            cfd_applied_map[idx][col] = None
    for i in range(0, len(cfd_list)):
        #stringified_cfd = '(' + cfd.lhs + ') => ' + cfd.rhs
        d_rep, cfd_applied_map = applyCfd(project_id, d_rep, cfd_list[i], cfd_id_list[i], cfd_applied_map, current_iter)
        #d_rep = applyCfd(project_id, d_rep, stringified_cfd, cfd.cfd_id, receiver)
    return d_rep, cfd_applied_map

def applyCfd(project_id, d_rep, cfd, cfd_id, cfd_applied_map, current_iter):
    #mod_count = 0
    tuple_metadata = pd.read_pickle('./store/' + project_id + '/tuple_metadata.p')
    for idx, row in d_rep.iterrows():
        #mod_count = 0
        if row['cover'] is not None and cfd in row['cover'].split('; '):
            lhs = cfd.split(' => ')[0][1:-1]
            rhs = cfd.split(' => ')[1]
            if '=' in rhs:
                rh = np.array(rhs.split('='))
                if row[rh[0]] != rh[1]:
                    row[rh[0]] = rh[1]
                    #mod_count += 1
                    cfd_applied_map[idx][rh[0]] = cfd_id
        #tuple_metadata.at[idx, 'weight'] += mod_count
        #charm.reinforce(receiver, cfd_id, 1)
    return d_rep, cfd_applied_map


def reinforceTuplesBasedOnContradiction(project_id, current_iter, d_latest, cfd_applied_map):
    tuple_metadata = pd.read_pickle('./store/' + project_id + '/tuple_metadata.p')
    value_metadata = pickle.load( open('./store/' + project_id + '/value_metadata.p', 'rb') )
    #prev_iter = "{:08x}".format(int('0x' + current_iter, 0) - 1)
    for idx in d_latest.index:
        reinforcementValue = 0
        for col in d_latest.columns:
            prev_spread = len(set(value_metadata[idx][col]['history']))
            value_metadata[idx][col]['history'].append((d_latest.at[idx, col], cfd_applied_map[idx][col]))
            curr_spread = len(set(value_metadata[idx][col]['history']))
            vspr_d = 0       # value spread delta (change in value spread from last iteration)
            if curr_spread > prev_spread:
                vspr_d = 1

            cell_values = Counter([val for (val, cfd) in value_metadata[idx][col]['history']])
            num_occurrences_mode = cell_values.most_common(1)[0][1]
            new_vdis = 1 - (num_occurrences_mode/len(value_metadata[idx][col]['history']))   # new value disagreement (value disagreement for the current iteration)
            vdis_d = new_vdis - value_metadata[idx][col]['disagreement']            # value disagreement delta (change in value disagreement from last iteration; set to 0 if less than 0 so as to disallow negative tuple weights)
            value_metadata[idx][col]['disagreement'] = new_vdis
            if vdis_d < 0:
                vdis_d = 0

            reinforcementValue += (vspr_d + new_vdis + vdis_d)     # add change in value spread, new value disagreement, and change in value disagreement from last iteration

        tuple_metadata.at[idx, 'weight'] += reinforcementValue

    #tuple_weights['weight'] = tuple_weights['weight']/tuple_weights['weight'].sum()
    print('Tuple weights:')
    pprint(tuple_metadata['weight'])
    print()
    tuple_metadata.to_pickle('./store/' + project_id + '/tuple_metadata.p')
    pickle.dump( value_metadata, open('./store/' + project_id + '/value_metadata.p', 'wb') )

def reinforceCFDs(project_id, cfd_id, receiver, cfd_metadata):
    return None


# TODO
def buildSample(d_rep, sample_size, project_id, cfd_applied_map, current_iter):
    # TEMPORARY IMPLEMENTATION
    #tuple_metadata = pd.read_pickle('./store/' + project_id + '/tuple_metadata.p')
    cfd_metadata = pickle.load( open('./store/' + project_id + '/cfd_metadata.p', 'rb') )
    chosen_tups = tuple_weights.sample(n=sample_size, weights='weight')     # tuples with a larger weight (a.k.a. larger value in the 'weight' column of tuple_weights) are more likely to be chosen
    print('Chosen example indexes:')
    pprint(chosen_tups.index)

    for idx in chosen_tups.index:
        seen = list()
        for col in d_rep.columns:
        # TODO; Calculate how many rows the CFD was applied to in the sample
            if cfd_applied_map[idx][col] not in seen:
                seen.append(cfd_applied_map[idx][col])
                cfd_metadata[cfd_applied_map[idx][col]]['num_changes'][current_iter] += 1

    pickle.dump(cfd_metadata, open('./store/' + project_id + '/cfd_metadata.p', 'wb') )
    sample = d_rep.iloc[chosen_tups.index]
    return sample
