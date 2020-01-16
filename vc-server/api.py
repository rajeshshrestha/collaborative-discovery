from flask import Flask, request, send_file
from flask_restful import Resource, Api, reqparse, abort
from flask_cors import CORS
from flask_csv import send_csv
from random import sample
from pprint import pprint
from zipfile import ZipFile
from io import BytesIO
import json
import os
import subprocess
import helpers
import time
import pandas as pd
import numpy as np
import pickle
import math

app = Flask(__name__)
CORS(app)
api = Api(app)


class Import(Resource):
    def get(self):
        return {'msg': '[SUCCESS] Import test success!'}

    def post(self):
        newProjectID = 0
        existingProjects = [('0x' + d) for d in os.listdir('./store/') if os.path.isdir(os.path.join('./store/', d))]
        newDir = ''
        if len(existingProjects) == 0:
            newProjectID = "{:08x}".format(1)
        else:
            projectIDList = [int(d, 0) for d in existingProjects]
            print(projectIDList)
            newProjectID = "{:08x}".format(max(projectIDList) + 1)
            print(newProjectID)
        newDir = './store/' + newProjectID + '/'
        try:
            os.mkdir(newDir)
            newDir += '00000001/'
            os.mkdir(newDir)
        except OSError:
            print ('[ERROR] Unable to create a new directory for the project.')
            return {'msg': '[ERROR] Unable to create a new directory for the project.'}
        importedFile = request.files['file']
        f = open(newDir + 'data.csv', 'w')
        data = importedFile.read().decode('utf-8-sig').split('\n')
        header = data[0].split(',')
        for line in [l for l in data if len(l) > 0]:
            trimmedLineList = [tL.strip() for tL in line.split(',')]
            trimmedLine = ','.join(trimmedLineList)
            f.write(trimmedLine + '\n')
        f.close()

        #df = pd.read_csv(newDir + 'data.csv')
        #for idx in df.index:
        #    for col in df.columns:
        #        try:
        #            if math.isnan(float(df.at[idx, col])):
        #                df.at[idx, col] = ''
        #        except ValueError:
        #            pass
        #        except TypeError:
        #            pass

        #df.to_csv(newDir + 'data.csv', encoding='utf-8', index=False)

        returned_data = {
            'header': header,
            'project_id': newProjectID,
            'msg': '[SUCCESS] Successfully created new project with project ID = ' + newProjectID + '.'
        }
        response = json.dumps(returned_data)
        pprint(response)
        return response, 200


class Sample(Resource):
    def get(self):
        return {'msg': '[SUCCESS] Sample test success!'}

    def post(self):
        print(request.form)
        print(request.form.get('project_id'))
        project_id = request.form.get('project_id')
        sample_size = int(request.form.get('sample_size'))
        existing_iters = [('0x' + f) for f in os.listdir('./store/' + project_id + '/') if os.path.isdir(os.path.join('./store/' + project_id + '/', f))]
        iteration_list = [int(d, 0) for d in existing_iters]
        print(iteration_list)
        current_iter = "{:08x}".format(max(iteration_list))
        print(current_iter)
        data = pd.read_csv('./store/' + project_id + '/' + current_iter + '/data.csv', keep_default_na=False)
        tuple_weights = pd.DataFrame(index=data.index, columns=['weight'])
        tuple_weights['weight'] = 1
        exploration_freq = pd.DataFrame(index=data.index, columns=['count'])
        exploration_freq['count'] = 0
        value_mapper = dict()
        value_spread = dict()
        value_disagreement = dict()
        print(tuple_weights)
        print(value_mapper)
        for idx in data.index:
            value_mapper[idx] = dict()
            value_spread[idx] = dict()
            value_disagreement[idx] = dict()
            for col in data.columns:
                print(idx, col)
                #value_mapper.at[idx, col] = ' '.join([str(data.at[idx, col])])
                value_mapper[idx][col] = [data.at[idx, col]]
                value_spread[idx][col] = 1
                value_disagreement[idx][col] = 0
        #value_mapper.to_pickle('./store/' + project_id + '/value_mapper.p')
        pickle.dump( value_mapper, open('./store/' + project_id + '/value_mapper.p', 'wb') )
        pickle.dump( value_spread, open('./store/' + project_id + '/00000001/value_spread.p', 'wb') )
        pickle.dump( value_disagreement, open('./store/' + project_id + '/00000001/value_disagreement.p', 'wb') )
        s_out = helpers.buildSample(data, min(sample_size, len(data.index)), project_id)   # SAMPLING FUNCTION GOES HERE; FOR NOW, BASIC SAMPLER
        for idx in data.index:
            if idx in s_out.index:
                exploration_freq.at[idx, 'count'] += 1
            else:
                tuple_weights.at[idx, 'count'] += 1

        tuple_weights['weight'] = tuple_weights['weight'] / tuple_weights['weight'].sum()
        tuple_weights.to_pickle('./store/' + project_id + '/tuple_weights.p')
        exploration_freq.to_pickle('./store/' + project_id + '/exploration_freq.p')

        returned_data = {
            'sample': s_out.to_json(orient='index'),
            'msg': '[SUCCESS] Successfully retrieved sample.'
        }
        response = json.dumps(returned_data)
        pprint(response)
        return response, 200


class Clean(Resource):
    def get(self):
        return {'msg': '[SUCCESS] Clean test success!'}

    def post(self):
        print(request.form)
        print(request.form.get('project_id'))
        print(request.form.get('data'))
        print(request.form.get('sample_size'))

        print('About to read form data')

        project_id = request.form.get('project_id')
        s_in = request.form.get('data')
        sample_size = int(request.form.get('sample_size'))

        print('Read form data')

        existing_iters = [('0x' + f) for f in os.listdir('./store/' + project_id + '/') if os.path.isdir(os.path.join('./store/' + project_id + '/', f))]
        iteration_list = [int(d, 0) for d in existing_iters]
        current_iter = "{:08x}".format(max(iteration_list) + 1)
        print("New iteration: " + str(current_iter))
        prev_iter = "{:08x}".format(current_iter - 1)

        d_dirty = pd.read_csv('./store/' + project_id + '/' + prev_iter + '/data.csv', keep_default_na=False)
        d_rep = helpers.applyUserRepairs(d_dirty, s_in)
        os.mkdir('./store/' + project_id + '/' + current_iter + '/')
        d_rep.to_csv('./store/' + project_id + '/' + current_iter + '/data.csv', encoding='utf-8', index=False)
        print('about to discover CFDs')
        top_cfds = helpers.discoverCFDs(project_id, current_iter)
        d_rep['cover'] = None

        if top_cfds is not None and isinstance(top_cfds, np.ndarray):
            helpers.addNewCfdsToList(top_cfds, project_id)
            #receiver = helpers.addNewCfdsToList(top_cfds, project_id) # TODO; this will eventually be the function call used for addNewCfdsToList

            d_rep = helpers.buildCover(d_rep, top_cfds)

            picked_cfd_list = helpers.pickCfds(top_cfds, 1)      # TODO; will eventually use charmPickCfds instead

            # TODO: everything through the "pickle.dump" line will eventually be outside of this if statement, once Charm is integrated
            #picked_cfd_list, picked_idx_list = helpers.charmPickCfds(receiver, query, sample_size)

            if picked_cfd_list is not None:
                np.savetxt('./store/' + project_id + '/' + current_iter + '/applied_cfds.txt', picked_cfd_list,
                           fmt="%s")
                d_rep = helpers.applyCfdList(project_id, d_rep, picked_cfd_list)
                #d_rep = helpers.applyCfdList(d_rep, picked_cfd_list, picked_idx_list)  # TODO: This will eventually be the function call used for applyCfdList
            else:
                with open('./store/' + project_id + '/' + current_iter + '/applied_cfds.txt', 'w') as f:
                    print('No CFDs were applied.', file=f)
            #pickle.dump( receiver, open('./store/' + project_id + '/charm_receiver.p', 'wb') )     # TODO: uncomment to save receiver into pickle file

        d_rep = d_rep.drop(columns=['cover'])
        helpers.reinforceTuplesBasedOnContradiction(project_id, current_iter, d_rep)
        d_rep.to_csv('./store/' + project_id + '/' + current_iter + '/data.csv', encoding='utf-8', index=False)
        s_out = helpers.buildSample(d_rep, sample_size, project_id)     # TODO; TEMPORARY IMPLEMENTATION

        tuple_weights = pd.read_pickle('./store/' + project_id + '/tuple_weights.p')
        exploration_freq = pd.read_pickle('./store/' + project_id + 'exploration_freq.p')

        for idx in d_rep.index:
            if idx in s_out.index:
                exploration_freq.at[idx, 'count'] += 1
            else:
                tuple_weights.at[idx, 'count'] += (1 - (exploration_freq.at[idx, 'count']/int(current_iter, 0)))    # reinforce tuple based on how frequently been explored

        tuple_weights['weight'] = tuple_weights['weight'] / tuple_weights['weight'].sum()
        tuple_weights.to_pickle('./store/' + project_id + '/tuple_weights.p')
        exploration_freq.to_pickle('./store/' + project_id + '/exploration_freq.p')

        returned_data = {
            'sample': s_out.to_json(orient='index'),
            'msg': '[SUCCESS] Successfully applied and generalized repair and retrived new sample.'
        }
        response = json.dumps(returned_data)
        pprint(response)
        return response, 200


class Download(Resource):
    def get(self):
        return {'msg': '[SUCCESS] Result test success!'}

    def post(self):
        print(request.form)
        print(request.form.get('project_id'))
        project_id = request.form.get('project_id')

        existing_iters = [('0x' + f) for f in os.listdir('./store/' + project_id + '/') if
                          os.path.isdir(os.path.join('./store/' + project_id + '/', f))]
        iteration_list = [int(d, 0) for d in existing_iters]
        print(iteration_list)
        latest_iter = "{:08x}".format(max(iteration_list))
        print(latest_iter)

        finalZip = BytesIO()

        with ZipFile(finalZip, 'w') as zf:
            zf.write('./store/' + project_id + '/' + latest_iter + '/data.csv')
            zf.write('./store/' + project_id + '/' + latest_iter + '/applied_cfds.txt')
        finalZip.seek(0)

        return send_file(finalZip, attachment_filename='charm_cleaned.zip', as_attachment=True)


api.add_resource(Import, '/import')
api.add_resource(Sample, '/sample')
api.add_resource(Clean, '/clean')
api.add_resource(Download, '/download')

if __name__ == '__main__':
    app.run(debug=True)
