from flask import Flask, request, send_file, jsonify
from flask_restful import Resource, Api, reqparse, abort
from flask_cors import CORS, cross_origin
from flask_csv import send_csv

import random
from pprint import pprint
import json
import os
import helpers
import time
import pandas as pd
import numpy as np
import pickle
import math
import shutil
import logging
import csv
import analyze

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

logging.getLogger('flask_cors').level = logging.DEBUG

class Test(Resource):
    def get(self):
        return {'msg': '[SUCCESS] The server is live!'}

class Import(Resource):
    def get(self):
        return {'msg': '[SUCCESS] This endpoint is live!'}

    def post(self):
        
        # Initialize a new project
        projects = [('0x' + d) for d in os.listdir('./store') if os.path.isdir(os.path.join('./store/', d))]
        sampling_methods = ['DUO', 'RANDOM-PURE']
        if len(projects) == 0:
            sampling_method = np.random.choice(sampling_methods)
            new_project_id = '{:08x}'.format(1)
        else:
            sampling_method_distribution = list()
            project_ids = [int(d, 0) for d in projects]
            for project in projects:
                with open('./store/' + project[2:] + '/project_info.json', 'r') as f:
                    project_id_info = json.load(f)
                sampling_method_distribution.append(project_id_info['scenario']['sampling_method'])
            duo_freq = 1 - len([s for s in sampling_method_distribution if s == 'DUO'])/len(sampling_method_distribution)
            rp_freq = 1 - len([s for s in sampling_method_distribution if s == 'RANDOM-PURE'])/len(sampling_method_distribution)
            freqs = [duo_freq, rp_freq]
            print(freqs)
            sampling_method = np.random.choice(sampling_methods, p=freqs)
            new_project_id = '{:08x}'.format(max(project_ids) + 1)
        new_project_dir = './store/' + new_project_id
        try:
            os.mkdir(new_project_dir)
        except OSError:
            returned_data = {
                'msg': '[ERROR] Unable to create a directory for this project.'
            }
            pprint(returned_data)
            response = json.dumps(returned_data)
            return response, 500, {'Access-Control-Allow-Origin': '*'}

        print("*** Project initialized ***")

        # Read the scenario number and initialize the scenario accordingly
        scenario_id = request.form.get('scenario_id')
        with open('scenarios.json', 'r') as f:
            scenarios_list = json.load(f)
        scenario = scenarios_list[scenario_id]
        scenario['sampling_method'] = sampling_method
        project_info = {
            'scenario_id': scenario_id,
            'scenario': scenario,
            'score': 0,
            'true_pos': 0,
            'false_pos': 0
        }
        with open(new_project_dir + '/project_info.json', 'w') as f:
            json.dump(project_info, f, indent=4)

        print('*** Project info saved ***')

        # Extract the header
        with open(scenario['dirty_dataset']) as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            data = list(reader)

        # Initialize iteration counter
        current_iter = 0

        # Initialize tuple metadata and value metadata objects
        tuple_metadata = dict()
        cell_metadata = dict()
        for idx in range(0, len(data)):

            # Tuple metadata
            tuple_metadata[idx] = dict()
            tuple_metadata[idx]['weight'] = 1/len(data)

            # Cell metadata
            cell_metadata[idx] = dict()
            for col in header:
                cell_metadata[idx][col] = dict()
                cell_metadata[idx][col]['feedback_history'] = list()
                
        # FD metadata
        cfd_metadata = dict()
        # data = pd.read_csv(scenario['dirty_dataset'], keep_default_na=False)
        for c in scenario['hypothesis_space']:
            cfd_metadata[c['cfd']] = dict()
            cfd_metadata[c['cfd']]['weight'] = c['conf']
            cfd_metadata[c['cfd']]['conf_history'] = list()
            cfd_metadata[c['cfd']]['conf_history'].append(helpers.CFDScore(iter_num=current_iter, score=c['conf'], elapsed_time=0))

            hypothesis = next(x for x in scenario['hypothesis_space'] if x['cfd'] == c['cfd'])
            cfd_metadata[c['cfd']]['support'] = hypothesis['support']
            cfd_metadata[c['cfd']]['vios'] = hypothesis['vios']
            cfd_metadata[c['cfd']]['vio_pairs'] = hypothesis['vio_pairs']
        
        # cfd_metadata = helpers.normalizeWeights(cfd_metadata)
        # for _, cfd_m in cfd_metadata.items():
            cfd_metadata[c['cfd']]['weight_history'] = list()
            cfd_metadata[c['cfd']]['weight_history'].append(helpers.CFDWeightHistory(iter_num=current_iter, weight=cfd_metadata[c['cfd']]['weight'], elapsed_time=0))

        # Initialize other metrics/metadata needed in study
        study_metrics = dict()
        study_metrics['true_error_pct_full'] = list()
        study_metrics['true_error_pct_iter'] = list()
        study_metrics['false_positives_full'] = list()
        study_metrics['false_positives_iter'] = list()
        study_metrics['cfd_confidence'] = dict()
        # for cfd in scenario['cfds']:
        #     study_metrics['cfd_confidence'][cfd] = list()
        study_metrics['error_accuracy_full'] = list()
        study_metrics['error_accuracy_iter'] = list()

        start_time = time.time()

        modeling_metadata = dict()
        modeling_metadata['p_X_given_h'] = dict()
        modeling_metadata['X'] = list()
        modeling_metadata['Y'] = list()
        modeling_metadata['y_in_h'] = dict()  # TODO: Update y_supp_h to y_in_h (y satisfies h or is a detectable violation of h)
        for cfd in cfd_metadata.keys():
            modeling_metadata['y_in_h'][cfd] = dict()

        gt_metadata = modeling_metadata
        clean_fd_metadata = cfd_metadata

        '''for c in scenario['clean_hypothesis_space']:
            if c['cfd'] not in cfd_metadata.keys():
                clean_fd_metadata[c['cfd']] = dict()
                clean_fd_metadata[c['cfd']]['weight'] = c['conf']
                clean_fd_metadata[c['cfd']]['conf_history'] = list()
                clean_fd_metadata[c['cfd']]['conf_history'].append(helpers.CFDScore(iter_num=current_iter, score=c['conf'], elapsed_time=0))

                hypothesis = next(x for x in scenario['clean_hypothesis_space'] if x['cfd'] == c['cfd'])
                clean_fd_metadata[c['cfd']]['support'] = hypothesis['support']
                clean_fd_metadata[c['cfd']]['vios'] = hypothesis['vios']
                clean_fd_metadata[c['cfd']]['vio_pairs'] = hypothesis['vio_pairs']
                clean_fd_metadata[c['cfd']]['weight_history'] = list()
                clean_fd_metadata[c['cfd']]['weight_history'].append(helpers.CFDWeightHistory(iter_num=current_iter, weight=clean_fd_metadata[c['cfd']]['weight'], elapsed_time=0))
                gt_metadata['y_in_h'][c['cfd']] = dict()'''

        # modeling_metadata['p_h'] = dict()
        # modeling_metadata['p_h']['hUniform'] = dict()   # TODO: Do this for each p(h) heuristic

        print('*** Metadata and study metric objects initialized ***')

        # Save metadata
        pickle.dump( tuple_metadata, open(new_project_dir + '/tuple_metadata.p', 'wb') )
        pickle.dump( cell_metadata, open(new_project_dir + '/cell_metadata.p', 'wb') )
        pickle.dump( cfd_metadata, open(new_project_dir + '/cfd_metadata.p', 'wb') )
        pickle.dump( current_iter, open(new_project_dir + '/current_iter.p', 'wb') )
        pickle.dump( study_metrics, open(new_project_dir + '/study_metrics.p', 'wb') )
        pickle.dump( start_time, open(new_project_dir + '/start_time.p', 'wb') )
        pickle.dump( modeling_metadata, open(new_project_dir + '/modeling_metadata.p', 'wb') )

        pickle.dump( gt_metadata, open(new_project_dir + '/gt_metadata.p', 'wb') )
        pickle.dump( clean_fd_metadata, open(new_project_dir + '/clean_fd_metadata.p', 'wb') )

        print('*** Metadata and study metric objects saved ***')

        # Return information to the user
        returned_data = {
            'header': header,
            'project_id': new_project_id,
            'scenario_desc': scenario['description'],
            'msg': '[SUCCESS] Successfully created new project with project ID = ' + new_project_id + '.'
        }
        response = json.dumps(returned_data)
        # pprint(response)
        return response, 200, {'Access-Control-Allow-Origin': '*'}

class Sample(Resource):
    def get(self):
        return {'msg': '[SUCCESS] This endpoint is live!'}

    def post(self):
        project_id = request.form.get('project_id')
        sample_size = 10
        with open('./store/' + project_id + '/project_info.json') as f:
            project_info = json.load(f)

        current_time = time.time()
        pickle.dump( current_time, open('./store/' + project_id + '/current_time.p', 'wb') )

        print('*** Project info loaded ***')
            
        data = pd.read_csv(project_info['scenario']['dirty_dataset'], keep_default_na=False)
        sampling_method = project_info['scenario']['sampling_method']
        
        # Build sample and update tuple weights post-sampling
        start_time = pickle.load( open('./store/' + project_id + '/start_time.p', 'rb') )
        current_iter = 0
        s_out = helpers.buildSample(data, sample_size, project_id, sampling_method, current_iter, start_time) # TODO: Update sampling strategy
        s_index = s_out.index
        pickle.dump( s_index, open('./store/' + project_id + '/current_sample.p', 'wb') )

        print('*** Sample built and saved with ' + sampling_method + ' ***')

        # Build initial feedback map for frontend
        feedback = list()
        for idx in s_out.index:
            for col in s_out.columns:
                feedback.append({
                    'row': idx,
                    'col': col,
                    'marked': False
                })

        print('*** Feedback object created ***')
        
        s_out.insert(0, 'id', s_out.index, True)

        # Return information to the user
        returned_data = {
            'sample': s_out.to_json(orient='index'),
            'feedback': json.dumps(feedback),
            'true_pos': 0,
            'false_pos': 0,
            'msg': '[SUCCESS] Successfully built sample.'
        }
        # pprint(returned_data)
        response = json.dumps(returned_data)
        return response, 200, {'Access-Control-Allow-Origin': '*'}

class Resume(Resource):
    def get(self):
        return {'msg': '[SUCCESS] This endpoint is live!'}

    def post(self):
        project_id = request.form.get('project_id')
        projects = [d for d in os.listdir('./store') if os.path.isdir(os.path.join('./store/', d))]
        if project_id not in projects:
            returned_data = {
                'msg': '[INVALID PROJECT ID]'
            }
            pprint(returned_data)
            response = json.dumps(returned_data)
            return response, 200, {'Access-Control-Allow-Origin': '*'}
            
        s_index = pickle.load( open('./store/' + project_id + '/current_sample.p', 'rb') )

        with open('./store/' + project_id + '/project_info.json') as f:
            project_info = json.load(f)

        data = pd.read_csv(project_info['scenario']['dirty_dataset'], keep_default_na=False)
        s_out = data.loc[s_index, :]

        # Build changes map for front-end
        feedback = list()
        cell_metadata = pickle.load( open('./store/' + project_id + '/cell_metadata.p', 'rb') )
        for idx in s_out.index:
            for col in s_out.columns:
                feedback.append({
                    'row': idx,
                    'col': col,
                    'marked': bool(cell_metadata[int(idx)][col]['feedback_history'][-1].marked) if len(cell_metadata[int(idx)][col]['feedback_history']) > 0 else False
                })

        print('*** Feedback object created ***')

        true_pos, false_pos = helpers.getUserScores(project_id)
        
        print('*** Leaderboard created ***')

        current_iter = pickle.load( open('./store/' + project_id + '/current_iter.p', 'rb') )

        h_space_conf_delta = helpers.getHSpaceConfDelta(project_id, current_iter)

        if current_iter >= 25 or (h_space_conf_delta is not None and h_space_conf_delta < 0.01):
            msg = '[DONE]'
        else:
            msg = '[SUCCESS]: Saved feedback and built new sample.'

        header = s_out.columns.tolist()
        s_out.insert(0, 'id', s_out.index, True)

        # Return information to the user
        returned_data = {
            'header': header,
            'sample': s_out.to_json(orient='index'),
            'feedback': json.dumps(feedback),
            'true_pos': true_pos,
            'false_pos': false_pos,
            'msg': msg,
            'scenario_id': project_info['scenario_id'],
            'scenario_desc': project_info['scenario']['description']
        }
        # pprint(returned_data)
        response = json.dumps(returned_data)
        return response, 200, {'Access-Control-Allow-Origin': '*'}

class Clean(Resource):
    def get(self):
        return {'msg': '[SUCCESS] This endpoint is live!'}

    def post(self):
        project_id = request.form.get('project_id')
        feedback = json.loads(request.form.get('feedback'))
        is_new_feedback = int(request.form.get('is_new_feedback'))
        refresh = int(request.form.get('refresh'))
        feedback = pd.DataFrame.from_dict(feedback, orient='index')
        print(feedback)
        sample_size = 10

        print('*** Necessary objects loaded ***')

        current_iter = pickle.load( open('./store/' + project_id + '/current_iter.p', 'rb') )
        current_iter += 1

        current_time = pickle.load( open('./store/' + project_id + '/current_time.p', 'rb') )
        current_time = time.time()

        print('*** Iteration counter updated ***')

        with open('./store/' + project_id + '/project_info.json', 'r') as f:
            project_info = json.load(f)

        print('*** Project info loaded ***')
        
        data = pd.read_csv(project_info['scenario']['dirty_dataset'], keep_default_na=False)

        print('*** Loaded dirty dataset ***')

        # Save noise feedback
        # percentage_errors_found = 0
        if is_new_feedback == 1 and refresh == 0:
            print('*** NEW FEEDBACK! ***')
            helpers.saveNoiseFeedback(data, feedback, project_id, current_iter, current_time)
            print('*** Noise feedback saved ***')
            s_in = data.iloc[feedback.index]
            print('*** Extracted sample from dataset ***')
            helpers.explainFeedback(data, s_in, project_id, current_iter, current_time)
            print('*** Mining completed and FD/CFD weights updated ***')
        
        # Update tuple weights pre-sampling
        sampling_method = project_info['scenario']['sampling_method']
        print('*** Sampling method retrieved ***')
        # if refresh == 0:
        #     helpers.reinforceTuplesBasedOnDependencies(data, project_id, current_iter, is_new_feedback, project_info)
        # print('*** Tuples reinforced based on FD/CFD weights ***')

        # Build sample
        s_out = helpers.buildSample(data, sample_size, project_id, sampling_method, current_iter, current_time)
        s_index = s_out.index
        pickle.dump( s_index, open('./store/' + project_id + '/current_sample.p', 'wb') )

        print('*** New sample created and saved ***')

        # Build feedback map for front-end
        feedback = list()
        cell_metadata = pickle.load( open('./store/' + project_id + '/cell_metadata.p', 'rb') )
        for idx in s_out.index:
            for col in s_out.columns:
                feedback.append({
                    'row': idx,
                    'col': col,
                    'marked': bool(cell_metadata[int(idx)][col]['feedback_history'][-1].marked) if len(cell_metadata[int(idx)][col]['feedback_history']) > 0 else False
                })

        print('*** Feedback object created ***')

        true_pos, false_pos = helpers.getUserScores(project_id)

        print('*** User scores retrieved ***')

        h_space_conf_delta = helpers.getHSpaceConfDelta(project_id, current_iter)

        if refresh == 0 and (current_iter >= 25 or (h_space_conf_delta is not None and h_space_conf_delta < 0.01)):
            msg = '[DONE]'
        else:
            msg = '[SUCCESS]: Saved feedback and built new sample.'

        s_out.insert(0, 'id', s_out.index, True)

        pickle.dump( current_iter, open('./store/' + project_id + '/current_iter.p', 'wb') )
        pickle.dump( current_time, open('./store/' + project_id + '/current_time.p', 'wb') )
        
        # Return information to the user
        returned_data = {
            'sample': s_out.to_json(orient='index'),
            'feedback': json.dumps(feedback),
            'true_pos': true_pos,
            'false_pos': false_pos,
            'msg': msg
        }
        # pprint(returned_data)
        response = json.dumps(returned_data)
        return response, 200, {'Access-Control-Allow-Origin': '*'}

api.add_resource(Test, '/duo/api/')
api.add_resource(Import, '/duo/api/import')
api.add_resource(Sample, '/duo/api/sample')
api.add_resource(Resume, '/duo/api/resume')
api.add_resource(Clean, '/duo/api/clean')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
