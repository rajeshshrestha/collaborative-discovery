import copy
import pandas as pd
import numpy as np
import scipy as sp
import sys
import helpers
import json
import matplotlib.pyplot as plt
from rich.console import Console
import os
import statstests

console = Console()

def eval_user_h(project_id, run_type):
    pathstart = './docker-out/' if run_type == 'real' else './store/'
    with open(pathstart + project_id + '/project_info.json', 'r') as f:
        project_info = json.load(f)
    scenario = project_info['scenario']
    scenario_id = project_info['scenario_id']
    data = pd.read_csv(scenario['dirty_dataset'], keep_default_na=False)
    clean_data = pd.read_csv(scenario['clean_dataset'], keep_default_na=False)
    target_fd = scenario['target_fd']
    h_space = scenario['hypothesis_space']
    with open(pathstart + project_id + '/interaction_metadata.json', 'r') as f:
        interaction_metadata = json.load(f)
    with open(pathstart + project_id + '/fd_metadata.json', 'r') as f:
        fd_metadata = json.load(f)
    with open(pathstart + project_id + '/study_metrics.json', 'r') as f:
        study_metrics = json.load(f)
    user_h_history = interaction_metadata['user_hypothesis_history']
    user_h_conf_history = list()
    fd_recall_history = list()
    fd_precision_history = list()
    fd_recall_seen_history = list()
    fd_precision_seen_history = list()
    user_h_seen_conf_history = list()
    seen_tuples = set()

    with open('./study-utils/users.json', 'r') as f:
        users = json.load(f)
    user_num_dict = dict()
    counter = 1
    for email in users.keys():
        user_num_dict[email] = counter
        counter += 1
    user_num = str(user_num_dict[project_info['email']])

    for h in user_h_history:
        fd = h['value'][0]
        if fd == 'Not Sure':
            user_h_conf_history.append(0)
            fd_recall_history.append(0)
            fd_precision_history.append(0)
            if h['iter_num'] > 0:
                user_h_seen_conf_history.append(0)
                fd_recall_seen_history.append(0)
                fd_precision_seen_history.append(0)
            continue

        lhs = fd.split(' => ')[0][1:-1].split(', ')
        rhs = fd.split(' => ')[1].split(', ')
        try:
            fd_meta = next(f for f in scenario['clean_hypothesis_space'] \
                if set(f['cfd'].split(' => ')[0][1:-1].split(', ')) == set(lhs) \
                and set(f['cfd'].split(' => ')[1].split(', ')) == set(rhs))
            dirty_fd_meta = next(f for f in scenario['hypothesis_space'] \
                if set(f['cfd'].split(' => ')[0][1:-1].split(', ')) == set(lhs) \
                and set(f['cfd'].split(' => ')[1].split(', ')) == set(rhs))
            support, vios = dirty_fd_meta['support'], dirty_fd_meta['vios']
            conf = fd_meta['conf']
            # fd = fd_meta['cfd']
        except StopIteration:
            support, vios = helpers.getSupportAndVios(data, clean_data, fd)
            conf = (len(support) - len(vios)) / len(support)
        
        target_fd_dirty_meta = next(f for f in scenario['hypothesis_space'] if f['cfd'] == scenario['target_fd'])
        target_vios = target_fd_dirty_meta['vios']

        user_h_conf_history.append(conf)
        fd_recall = len([v for v in vios if v in target_vios]) / len(target_vios)
        fd_precision = 0 if len(vios) == 0 else len([v for v in vios if v in target_vios]) / len(vios)
        fd_recall_history.append(fd_recall)
        fd_precision_history.append(fd_precision)

        if h['iter_num'] == 0:
            continue
        current_sample = next(i['value'] for i in interaction_metadata['sample_history'] if i['iter_num'] == h['iter_num'])
        seen_tuples |= set(current_sample)
        seen_data = data.iloc[list(seen_tuples)]
        seen_clean_data = clean_data.iloc[list(seen_tuples)]
        support, vios = helpers.getSupportAndVios(seen_data, seen_clean_data, fd)
        _, target_vios_seen = helpers.getSupportAndVios(seen_data, seen_clean_data, target_fd)
        conf = (len(support) - len(vios)) / len(support)
        user_h_seen_conf_history.append(conf)

        fd_recall_seen = len([v for v in vios if v in target_vios_seen]) / len(target_vios_seen)
        fd_precision_seen = 0 if len(vios) == 0 else len([v for v in vios if v in target_vios_seen]) / len(vios)

        fd_recall_seen_history.append(fd_recall_seen)
        fd_precision_seen_history.append(fd_precision_seen)
    
    study_metrics, fd_metadata = helpers.deriveStats(
        interaction_metadata,
        fd_metadata,
        h_space,
        study_metrics,
        data,
        clean_data,
        target_fd
    )
    with open(pathstart + project_id + '/study_metrics.json', 'w') as f:
        json.dump(study_metrics, f)
    with open(pathstart + project_id + '/fd_metadata.json', 'w') as f:
        json.dump(fd_metadata, f)
    
    cumulative_precision, cumulative_recall = study_metrics['cumulative_precision'], study_metrics['cumulative_recall']
    cumulative_precision_noover, cumulative_recall_noover = study_metrics['cumulative_precision_noover'], study_metrics['cumulative_recall_noover']
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    fig7, ax7 = plt.subplots()
    fig8, ax8 = plt.subplots()
    fig9, ax9 = plt.subplots()
    fig10, ax10 = plt.subplots()

    ax1.set_xticks(np.arange(0, 15, 3))
    ax2.set_xticks(np.arange(0, 15, 3))
    ax3.set_xticks(np.arange(0, 15, 3))
    ax4.set_xticks(np.arange(0, 15, 3))
    ax5.set_xticks(np.arange(0, 15, 3))
    ax6.set_xticks(np.arange(0, 15, 3))
    ax7.set_xticks(np.arange(0, 15, 3))
    ax8.set_xticks(np.arange(0, 15, 3))
    ax9.set_xticks(np.arange(0, 15, 3))
    ax10.set_xticks(np.arange(0, 15, 3))

    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])
    ax3.set_ylim([0, 1])
    ax4.set_ylim([0, 1])
    ax5.set_ylim([0, 1])
    ax6.set_ylim([0, 1])
    ax7.set_ylim([0, 1])
    ax8.set_ylim([0, 1])
    ax9.set_ylim([0, 1])
    ax10.set_ylim([0, 1])

    ax1.plot([i['iter_num'] for i in user_h_history], user_h_conf_history)
    data1 = pd.DataFrame(columns=['iter_num', 'fd_confidence'])
    data1['iter_num'] = [i['iter_num'] for i in user_h_history]
    data1['fd_confidence'] = user_h_conf_history
    statstests.mannkendall(data1)

    ax2.plot([i['iter_num'] for i in user_h_history], fd_recall_history)
    data2 = pd.DataFrame(columns=['iter_num', 'fd_recall'])
    data2['iter_num'] = [i['iter_num'] for i in user_h_history]
    data2['fd_recall'] = fd_recall_history
    statstests.mannkendall(data2)

    ax3.plot([i['iter_num'] for i in user_h_history if i['iter_num'] > 0], user_h_seen_conf_history)
    data3 = pd.DataFrame(columns=['iter_num', 'fd_confidence'])
    data3['iter_num'] = [i['iter_num'] for i in user_h_history]
    data3['fd_confidence'] = user_h_seen_conf_history
    statstests.mannkendall(data3)

    ax4.plot([i['iter_num'] for i in cumulative_precision], [i['value'] for i in cumulative_precision])
    data4 = pd.DataFrame(columns=['iter_num', 'user_precision'])
    data4['iter_num'] = [i['iter_num'] for i in cumulative_precision]
    data4['user_precision'] = [i['value'] for i in cumulative_precision]
    statstests.mannkendall(data4)

    ax5.plot([i['iter_num'] for i in cumulative_recall], [i['value'] for i in cumulative_recall])
    data5 = pd.DataFrame(columns=['iter_num', 'user_recall'])
    data5['iter_num'] = [i['iter_num'] for i in cumulative_recall]
    data5['user_recall'] = [i['value'] for i in cumulative_recall]
    statstests.mannkendall(data5)

    ax6.plot([i['iter_num'] for i in cumulative_precision_noover], [i['value'] for i in cumulative_precision_noover])
    data6 = pd.DataFrame(columns=['iter_num', 'user_precision'])
    data6['iter_num'] = [i['iter_num'] for i in cumulative_precision_noover]
    data6['user_precision'] = [i['value'] for i in cumulative_precision_noover]
    statstests.mannkendall(data6)

    ax7.plot([i['iter_num'] for i in cumulative_recall_noover], [i['value'] for i in cumulative_recall_noover])
    data7 = pd.DataFrame(columns=['iter_num', 'user_recall'])
    data7['iter_num'] = [i['iter_num'] for i in cumulative_recall_noover]
    data7['user_recall'] = [i['value'] for i in cumulative_recall_noover]
    statstests.mannkendall(data7)

    ax8.plot([i['iter_num'] for i in user_h_history], fd_precision_history)
    data8 = pd.DataFrame(columns=['iter_num', 'fd_precision'])
    data8['iter_num'] = [i['iter_num'] for i in user_h_history]
    data8['fd_precision'] = fd_precision_history
    statstests.mannkendall(data8)

    ax9.plot([i['iter_num'] for i in user_h_history if i['iter_num'] > 0], fd_recall_seen_history)
    data9 = pd.DataFrame(columns=['iter_num', 'fd_recall'])
    data9['iter_num'] = [i['iter_num'] for i in user_h_history]
    data9['fd_recall'] = fd_recall_seen_history
    statstests.mannkendall(data9)

    ax10.plot([i['iter_num'] for i in user_h_history if i['iter_num'] > 0], fd_precision_seen_history)
    data10 = pd.DataFrame(columns=['iter_num', 'fd_precision'])
    data10['iter_num'] = [i['iter_num'] for i in user_h_history]
    data10['fd_precision'] = fd_precision_seen_history
    statstests.mannkendall(data10)

    ax1.set_xlabel('Iteration #')
    ax1.set_ylabel('Confidence')
    ax1.set_title('Suggested FD Confidence Over the Interaction')
   
    ax2.set_xlabel('Iteration #')
    ax2.set_ylabel('Recall')
    ax2.set_title('Suggested FD Recall')
    
    ax3.set_xlabel('Iteration #')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Suggested FD Confidence Over What the User Has Seen')
    
    ax4.set_xlabel('Iteration #')
    ax4.set_ylabel('Precision')
    ax4.set_title('Cumulative User Precision')
    
    ax5.set_xlabel('Iteration #')
    ax5.set_ylabel('Recall')
    ax5.set_title('Cumulative User Recall')
    
    ax6.set_xlabel('Iteration #')
    ax6.set_ylabel('Precision')
    ax6.set_title('Cumulative User Precision (w/o Duplicate Vios)')
    
    ax7.set_xlabel('Iteration #')
    ax7.set_ylabel('Recall')
    ax7.set_title('Cumulative User Recall (w/o Duplicate Vios)')

    ax8.set_xlabel('Iteration #')
    ax8.set_ylabel('Precision')
    ax8.set_title('Suggested FD Precision')

    ax9.set_xlabel('Iteration #')
    ax9.set_ylabel('Recall')
    ax9.set_title('Suggested FD Recall Over What the User Has Seen')

    ax10.set_xlabel('Iteration #')
    ax10.set_ylabel('Precision')
    ax10.set_title('Suggested FD Precision Over What the User Has Seen')

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    fig6.tight_layout()
    fig7.tight_layout()
    fig8.tight_layout()
    fig9.tight_layout()
    fig10.tight_layout()

    fig1.savefig('./plots/fd-confidence/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    fig2.savefig('./plots/fd-recall/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    fig3.savefig('./plots/fd-confidence-seen/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    fig4.savefig('./plots/cumulative-user-precision/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    fig5.savefig('./plots/cumulative-user-recall/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    fig6.savefig('./plots/cumulative-user-precision-nodup/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    fig7.savefig('./plots/cumulative-user-recall-nodup/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    fig8.savefig('./plots/fd-precision/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    fig9.savefig('./plots/fd-recall-seen/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    fig10.savefig('./plots/fd-precision-seen/' + project_id + '-s' + scenario_id + '-u' + user_num + '.jpg')
    
    plt.clf()

def eval_h_grouped(group_type, run_type, id):
    pathstart = './docker-out/' if run_type == 'real' else './store/'

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    fig7, ax7 = plt.subplots()
    fig8, ax8 = plt.subplots()
    fig9, ax9 = plt.subplots()
    fig10, ax10 = plt.subplots()

    ax1.set_xticks(np.arange(0, 15, 3))
    ax2.set_xticks(np.arange(0, 15, 3))
    ax3.set_xticks(np.arange(0, 15, 3))
    ax4.set_xticks(np.arange(0, 15, 3))
    ax5.set_xticks(np.arange(0, 15, 3))
    ax6.set_xticks(np.arange(0, 15, 3))
    ax7.set_xticks(np.arange(0, 15, 3))
    ax8.set_xticks(np.arange(0, 15, 3))
    ax9.set_xticks(np.arange(0, 15, 3))
    ax10.set_xticks(np.arange(0, 15, 3))

    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])
    ax3.set_ylim([0, 1])
    ax4.set_ylim([0, 1])
    ax5.set_ylim([0, 1])
    ax6.set_ylim([0, 1])
    ax7.set_ylim([0, 1])
    ax8.set_ylim([0, 1])
    ax9.set_ylim([0, 1])
    ax10.set_ylim([0, 1])

    with open('./study-utils/users.json', 'r') as f:
        users = json.load(f)
    user_num_dict = dict()
    counter = 1
    for email in users.keys():
        user_num_dict[email] = counter
        counter += 1

    all_project_ids = os.listdir(pathstart)
    project_ids = list()
    for project_id in all_project_ids:
        with open(pathstart + project_id + '/project_info.json', 'r') as f:
            project_info = json.load(f)
        scenario = project_info['scenario']
        scenario_id = project_info['scenario_id']
        user_num = str(user_num_dict[project_info['email']])

        if group_type == 'scenario':
            if scenario_id != id:
                continue
        elif group_type == 'user':
            if int(id) not in user_num_dict.values():
                continue
        
        project_ids.append(project_id)
    
    for project_id in project_ids:
        with open(pathstart + project_id + '/project_info.json', 'r') as f:
            project_info = json.load(f)
        scenario = project_info['scenario']
        scenario_id = project_info['scenario_id']
        data = pd.read_csv(scenario['dirty_dataset'], keep_default_na=False)
        clean_data = pd.read_csv(scenario['clean_dataset'], keep_default_na=False)
        target_fd = scenario['target_fd']
        h_space = scenario['hypothesis_space']
        with open(pathstart + project_id + '/interaction_metadata.json', 'r') as f:
            interaction_metadata = json.load(f)
        with open(pathstart + project_id + '/fd_metadata.json', 'r') as f:
            fd_metadata = json.load(f)
        with open(pathstart + project_id + '/study_metrics.json', 'r') as f:
            study_metrics = json.load(f)
        user_h_history = interaction_metadata['user_hypothesis_history']
        user_h_conf_history = list()
        fd_recall_history = list()
        fd_precision_history = list()
        fd_recall_seen_history = list()
        fd_precision_seen_history = list()
        user_h_seen_conf_history = list()
        seen_tuples = set()

        for h in user_h_history:
            fd = h['value'][0]
            if fd == 'Not Sure':
                user_h_conf_history.append(0)
                fd_recall_history.append(0)
                fd_precision_history.append(0)
                if h['iter_num'] > 0:
                    user_h_seen_conf_history.append(0)
                    fd_recall_seen_history.append(0)
                    fd_precision_seen_history.append(0)
                continue

            lhs = fd.split(' => ')[0][1:-1].split(', ')
            rhs = fd.split(' => ')[1].split(', ')
            try:
                fd_meta = next(f for f in scenario['clean_hypothesis_space'] \
                    if set(f['cfd'].split(' => ')[0][1:-1].split(', ')) == set(lhs) \
                    and set(f['cfd'].split(' => ')[1].split(', ')) == set(rhs))
                dirty_fd_meta = next(f for f in scenario['hypothesis_space'] \
                    if set(f['cfd'].split(' => ')[0][1:-1].split(', ')) == set(lhs) \
                    and set(f['cfd'].split(' => ')[1].split(', ')) == set(rhs))
                support, vios = dirty_fd_meta['support'], dirty_fd_meta['vios']
                conf = fd_meta['conf']
                # fd = fd_meta['cfd']
            except StopIteration:
                support, vios = helpers.getSupportAndVios(data, clean_data, fd)
                conf = (len(support) - len(vios)) / len(support)
            
            target_fd_dirty_meta = next(f for f in scenario['hypothesis_space'] if f['cfd'] == scenario['target_fd'])
            target_vios = target_fd_dirty_meta['vios']

            user_h_conf_history.append(conf)
            fd_recall = len([v for v in vios if v in target_vios]) / len(target_vios)
            fd_precision = 0 if len(vios) == 0 else len([v for v in vios if v in target_vios]) / len(vios)
            fd_recall_history.append(fd_recall)
            fd_precision_history.append(fd_precision)

            if h['iter_num'] == 0:
                continue
            current_sample = next(i['value'] for i in interaction_metadata['sample_history'] if i['iter_num'] == h['iter_num'])
            seen_tuples |= set(current_sample)
            seen_data = data.iloc[list(seen_tuples)]
            seen_clean_data = clean_data.iloc[list(seen_tuples)]
            support, vios = helpers.getSupportAndVios(seen_data, seen_clean_data, fd)
            _, target_vios_seen = helpers.getSupportAndVios(seen_data, seen_clean_data, target_fd)
            conf = (len(support) - len(vios)) / len(support)
            user_h_seen_conf_history.append(conf)

            fd_recall_seen = len([v for v in vios if v in target_vios_seen]) / len(target_vios_seen)
            fd_precision_seen = 0 if len(vios) == 0 else len([v for v in vios if v in target_vios_seen]) / len(vios)

            fd_recall_seen_history.append(fd_recall_seen)
            fd_precision_seen_history.append(fd_precision_seen)
        
        study_metrics, fd_metadata = helpers.deriveStats(
            interaction_metadata,
            fd_metadata,
            h_space,
            study_metrics,
            data,
            clean_data,
            target_fd
        )
        with open(pathstart + project_id + '/study_metrics.json', 'w') as f:
            json.dump(study_metrics, f)
        with open(pathstart + project_id + '/fd_metadata.json', 'w') as f:
            json.dump(fd_metadata, f)
        
        cumulative_precision, cumulative_recall = study_metrics['cumulative_precision'], study_metrics['cumulative_recall']
        cumulative_precision_noover, cumulative_recall_noover = study_metrics['cumulative_precision_noover'], study_metrics['cumulative_recall_noover']

        ax1.plot([i['iter_num'] for i in user_h_history], user_h_conf_history)
        data1 = pd.DataFrame(columns=['iter_num', 'fd_confidence'])
        data1['iter_num'] = [i['iter_num'] for i in user_h_history]
        data1['fd_confidence'] = user_h_conf_history
        statstests.mannkendall(data1)

        ax2.plot([i['iter_num'] for i in user_h_history], fd_recall_history)
        data2 = pd.DataFrame(columns=['iter_num', 'fd_recall'])
        data2['iter_num'] = [i['iter_num'] for i in user_h_history]
        data2['fd_recall'] = fd_recall_history
        statstests.mannkendall(data2)

        ax3.plot([i['iter_num'] for i in user_h_history if i['iter_num'] > 0], user_h_seen_conf_history)
        data3 = pd.DataFrame(columns=['iter_num', 'fd_confidence'])
        data3['iter_num'] = [i['iter_num'] for i in user_h_history]
        data3['fd_confidence'] = user_h_seen_conf_history
        statstests.mannkendall(data3)

        ax4.plot([i['iter_num'] for i in cumulative_precision], [i['value'] for i in cumulative_precision])
        data4 = pd.DataFrame(columns=['iter_num', 'user_precision'])
        data4['iter_num'] = [i['iter_num'] for i in cumulative_precision]
        data4['user_precision'] = [i['value'] for i in cumulative_precision]
        statstests.mannkendall(data4)

        ax5.plot([i['iter_num'] for i in cumulative_recall], [i['value'] for i in cumulative_recall])
        data5 = pd.DataFrame(columns=['iter_num', 'user_recall'])
        data5['iter_num'] = [i['iter_num'] for i in cumulative_recall]
        data5['user_recall'] = [i['value'] for i in cumulative_recall]
        statstests.mannkendall(data5)

        ax6.plot([i['iter_num'] for i in cumulative_precision_noover], [i['value'] for i in cumulative_precision_noover])
        data6 = pd.DataFrame(columns=['iter_num', 'user_precision'])
        data6['iter_num'] = [i['iter_num'] for i in cumulative_precision_noover]
        data6['user_precision'] = [i['value'] for i in cumulative_precision_noover]
        statstests.mannkendall(data6)

        ax7.plot([i['iter_num'] for i in cumulative_recall_noover], [i['value'] for i in cumulative_recall_noover])
        data7 = pd.DataFrame(columns=['iter_num', 'user_recall'])
        data7['iter_num'] = [i['iter_num'] for i in cumulative_recall_noover]
        data7['user_recall'] = [i['value'] for i in cumulative_recall_noover]
        statstests.mannkendall(data7)

        ax8.plot([i['iter_num'] for i in user_h_history], fd_precision_history)
        data8 = pd.DataFrame(columns=['iter_num', 'fd_precision'])
        data8['iter_num'] = [i['iter_num'] for i in user_h_history]
        data8['fd_precision'] = fd_precision_history
        statstests.mannkendall(data8)

        ax9.plot([i['iter_num'] for i in user_h_history if i['iter_num'] > 0], fd_recall_seen_history)
        data9 = pd.DataFrame(columns=['iter_num', 'fd_recall'])
        data9['iter_num'] = [i['iter_num'] for i in user_h_history]
        data9['fd_recall'] = fd_recall_seen_history
        statstests.mannkendall(data9)

        ax10.plot([i['iter_num'] for i in user_h_history if i['iter_num'] > 0], fd_precision_seen_history)
        data10 = pd.DataFrame(columns=['iter_num', 'fd_precision'])
        data10['iter_num'] = [i['iter_num'] for i in user_h_history]
        data10['fd_precision'] = fd_precision_seen_history
        statstests.mannkendall(data10)

    ax1.set_xlabel('Iteration #')
    ax1.set_ylabel('Confidence')
    ax1.set_title('Suggested FD Confidence Over the Interaction')
   
    ax2.set_xlabel('Iteration #')
    ax2.set_ylabel('Recall')
    ax2.set_title('Suggested FD Recall')
    
    ax3.set_xlabel('Iteration #')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Suggested FD Confidence Over What the User Has Seen')
    
    ax4.set_xlabel('Iteration #')
    ax4.set_ylabel('Precision')
    ax4.set_title('Cumulative User Precision')
    
    ax5.set_xlabel('Iteration #')
    ax5.set_ylabel('Recall')
    ax5.set_title('Cumulative User Recall')
    
    ax6.set_xlabel('Iteration #')
    ax6.set_ylabel('Precision')
    ax6.set_title('Cumulative User Precision (w/o Duplicate Vios)')
    
    ax7.set_xlabel('Iteration #')
    ax7.set_ylabel('Recall')
    ax7.set_title('Cumulative User Recall (w/o Duplicate Vios)')

    ax8.set_xlabel('Iteration #')
    ax8.set_ylabel('Precision')
    ax8.set_title('Suggested FD Precision')

    ax9.set_xlabel('Iteration #')
    ax9.set_ylabel('Recall')
    ax9.set_title('Suggested FD Recall Over What the User Has Seen')

    ax10.set_xlabel('Iteration #')
    ax10.set_ylabel('Precision')
    ax10.set_title('Suggested FD Precision Over What the User Has Seen')

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    fig6.tight_layout()
    fig7.tight_layout()
    fig8.tight_layout()
    fig9.tight_layout()
    fig10.tight_layout()

    fig1.savefig('./plots/fd-confidence/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    fig2.savefig('./plots/fd-recall/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    fig3.savefig('./plots/fd-confidence-seen/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    fig4.savefig('./plots/cumulative-user-precision/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    fig5.savefig('./plots/cumulative-user-recall/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    fig6.savefig('./plots/cumulative-user-precision-nodup/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    fig7.savefig('./plots/cumulative-user-recall-nodup/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    fig8.savefig('./plots/fd-precision/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    fig9.savefig('./plots/fd-recall-seen/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    fig10.savefig('./plots/fd-precision-seen/' + (('s' + scenario_id) if group_type == 'scenario' else ('-u' + user_num)) + '.jpg')
    
    plt.clf()

if __name__ == '__main__':
    run_type = sys.argv[1]
    diff = sys.argv[2]
    id = sys.argv[3]
    if '0' in diff:
        eval_user_h(diff, run_type, id)
    else:
        eval_h_grouped(diff, run_type, id)
