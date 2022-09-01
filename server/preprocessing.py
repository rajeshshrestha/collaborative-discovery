import pandas as pd
import numpy as np
import json
import subprocess as sp
from tqdm import tqdm
import helpers
from rich.console import Console

console = Console()


def dataDiff(dirty_df, clean_df):
    diff = pd.DataFrame(columns=clean_df.columns)
    for row in clean_df.index:
        for col in clean_df.columns:
            diff.at[row, col] = dirty_df.at[row, col] == clean_df.at[row, col]
    return diff


if __name__ == '__main__':

    '''
        Load the scenarios-master.json file
        Sample Data: 
            {
            '0': {'alt_h': [],
            'clean_dataset': './data/toy.csv',
            'description': 'Identify errors in staff directory data.',
            'difficulty': 1,
            'dirty_dataset': './data/dirty_toy.csv',
            'target_fd': '(areacode) => state'},
            }

        # ! Target Hypothesis represents what FD the violations in the data are actually caused by and what FD should hold with the fewest violations in reality
    '''
    with open('scenarios-master.json', 'r') as f:
        scenarios = json.load(f)

    all_scenarios = dict()
    for s_id, scenario in tqdm(scenarios.items()):
        data = pd.read_csv(scenario['dirty_dataset'], keep_default_na=False)
        clean_data = pd.read_csv(
            scenario['clean_dataset'], keep_default_na=False)
        min_conf = 0.001
        max_ant = 3

        '''
            Run cfddiscovery program in dirty dataset and clean dataset.
            Refer to this https://github.com/j-r77/cfddiscovery for further details in the arguments
        '''
        process = sp.Popen(['./data/cfddiscovery/CFDD', scenario['dirty_dataset'], str(len(data.index)),
                           str(min_conf), str(max_ant)], stdout=sp.PIPE, stderr=sp.PIPE, env={'LANG': 'C++'})     # CFDD
        clean_process = sp.Popen(['./data/cfddiscovery/CFDD', scenario['clean_dataset'], str(len(data.index)), str(
            min_conf), str(max_ant)], stdout=sp.PIPE, stderr=sp.PIPE, env={'LANG': 'C++'})   # CFDD for clean h space

        res = process.communicate()
        clean_res = clean_process.communicate()
        if process.returncode == 0:
            '''
                Parse CFDs from the subprocess
                Final fds:
                    [   
                        {'cfd': '(areacode) => state'},
                        {'cfd': '(name) => zip, areacode, state'},
                        {'cfd': '(areacode, name) => zip, phone, state'},
                        {'cfd': '(zip, areacode) => phone, name'},
                        {'cfd': '(state) => zip, areacode, phone'},
                        {'cfd': '(areacode, state) => zip, phone'} ...
                    ]
            '''
            output = res[0].decode('latin_1').replace(',]', ']').replace(
                '\r', '').replace('\t', '').replace('\n', '')
            fds = [c['cfd'] for c in json.loads(output, strict=False)['cfds'] if '=' not in c['cfd'].split(
                ' => ')[0] and '=' not in c['cfd'].split(' => ')[1] and c['cfd'].split(' => ')[0] != '()']

            fds = helpers.buildCompositionSpace(
                fds, None, data, clean_data, min_conf, max_ant)
        else:
            fds = list()

        '''
            Similarly parse in the clean dataset
        '''
        if clean_process.returncode == 0:
            clean_output = res[0].decode('latin_1').replace(',]', ']').replace(
                '\r', '').replace('\t', '').replace('\n', '')
            # NOTE: THIS SHOULD BE CLEAN_OUTPUT IN THE LINE BELOW, NOT OUTPUT
            clean_fds = [c['cfd'] for c in json.loads(clean_output, strict=False)['cfds'] if '=' not in c['cfd'].split(
                ' => ')[0] and '=' not in c['cfd'].split(' => ')[1] and c['cfd'].split(' => ')[0] != '()']
            clean_fds = helpers.buildCompositionSpace(
                clean_fds, None, clean_data, None, min_conf, max_ant)
        else:
            clean_fds = list()

        '''
            Find the intersecting cfds in clean and dirty
            Sample intersecting_fds:
                ['(name) => areacode, zip, state, phone',
                '(zip, state, areacode) => name',
                '(state, phone, name) => zip',
                '(zip, phone, name) => areacode',
                ]
        '''
        intersecting_fds = list(set([f['cfd'] for f in fds]).intersection(
            set([c['cfd'] for c in clean_fds])))

        '''
            Iterate over the dirty dataset that are in the intersecting fds with the clean dataset
            Append all the hypothesis to h_space

            Sample h_space:
            [{'cfd': '(zip, state) => areacode',
                'conf': 1.0,
                'score': 1,
                'support': [0,
                            1,
                            2,
                            .......
                            32,
                            33,
                            34],
                'vio_pairs': [],
                'vios': []}]

        '''
        h_space = list()
        for fd in fds:
            if fd['cfd'] not in intersecting_fds:
                continue
            h = dict()
            '''
                Assign cfd to cfd in h
                Assibn score to be 1
            '''
            h['cfd'] = fd['cfd']
            h['score'] = 1

            '''
                Find and assign Support, violations and violation pairs
                Support: indices of all the tuples in the dirty dataset
            '''
            support, vios = helpers.getSupportAndVios(
                data, clean_data, h['cfd'])
            vio_pairs = helpers.getPairs(data, support, h['cfd'])
            h['conf'] = (len(support) - len(vios)) / len(support)
            h['support'] = support
            h['vios'] = vios
            h['vio_pairs'] = vio_pairs
            h_space.append(h)

        '''
            Iterate over all the clean fds which are in the intersecting fds with the dirty data
        '''
        clean_h_space = list()
        for fd in clean_fds:
            if fd['cfd'] not in intersecting_fds:
                continue
            h = dict()
            '''
                Assign cfd and score to be 1
            '''
            h['cfd'] = fd['cfd']
            h['score'] = 1

            '''
                Get Support and violations using the clean data only
                Compute conf to be the proportion of data to be following FD in the clean dataset
            '''
            # // Todo: Need to look inside the support and vios function how the two dfs are used
            support, vios = helpers.getSupportAndVios(
                clean_data, None, h['cfd'])
            h['conf'] = (len(support) - len(vios)) / len(support)
            # console.log(fd['cfd'])
            # console.log(vios)
            clean_h_space.append(h)

        # ! min_conf used in the generation of cfd for both the clean and dirty data
        scenario['min_conf'] = min_conf
        scenario['max_ant'] = max_ant  # ! used in the generation of dfd
        # ! info on the hypothesis space in dirty data
        scenario['hypothesis_space'] = h_space
        # ! info on the hypothesis space in clean data
        scenario['clean_hypothesis_space'] = clean_h_space
        # console.log([(h['cfd'], h['conf'])
        #             for h in scenario['hypothesis_space']])
        # console.log([(h['cfd'], h['conf'])
        #             for h in scenario['clean_hypothesis_space']])

        # ! Assign target_fd that is present in the hypothesis_space. The order of fd obtained from cfd could 
        # ! have been different than the target_fd assigned using the scenarios_master.json itself. Accounting for that.
        scenario['target_fd'] = next(f['cfd'] for f in scenario['hypothesis_space'] if set(f['cfd'].split(' => ')[0][1:-1].split(', ')) == set(
            scenario['target_fd'].split(' => ')[0][1:-1].split(', ')) and set(f['cfd'].split(' => ')[1].split(', ')) == set(scenario['target_fd'].split(' => ')[1].split(', ')))
    
        #! Same for the alt hypothesis as well. Accounting for the chance of change of order in the lhs and rhs of the fds
        formatted_alt_h = list()
        for alt_fd in scenario['alt_h']:
            fd = next(f['cfd'] for f in scenario['hypothesis_space'] if set(f['cfd'].split(' => ')[0][1:-1].split(', ')) == set(alt_fd.split(
                ' => ')[0][1:-1].split(', ')) and set(f['cfd'].split(' => ')[1].split(', ')) == set(alt_fd.split(' => ')[1].split(', ')))
            formatted_alt_h.append(fd)
        scenario['alt_h'] = formatted_alt_h

        clean_data = pd.read_csv(
            scenario['clean_dataset'], keep_default_na=False)

        '''
            Capture the cells in the clean data which has been retained. True being the same cell value for the dirty one at the index as the clean one
            Sample:
                diff = {
                        '0': {'areacode': True,
                            'name': True,
                            'phone': True,
                            'state': True,
                            'zip': True},
                        '1': {'areacode': True,
                            'name': True,
                            'phone': True,
                            'state': True,
                            'zip': True},
                            .....
                        }

        '''
        diff_df = dataDiff(data, clean_data)
        diff = json.loads(diff_df.to_json(orient='index'))
        scenario['diff'] = diff

        all_scenarios[s_id] = scenario
        all_scenarios[s_id]['sampling_method'] = 'DUO'
        all_scenarios[s_id]['update_method'] = 'BAYESIAN'

    with open('scenarios_debug.json', 'w') as f:
        json.dump(all_scenarios, f)
