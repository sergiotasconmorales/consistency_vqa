# Project:
#   VQA
# Description:
#   Script to generate file that contains image name, model answer, gt answer (not encoded)
# Author: 
#   Sergio Tascon-Morales

import os
import json
import pickle
import torch
import misc.io as io
from os.path import join as jp

config_name = 'config_479'
question_type = 'fovea'
path_dict_grades = '/home/sergio814/Documents/PhD/code/logs/dict_grade'
path_config = 'vqa/config/idrid_regions/single/'

def get_pred_ans(data, index):
    data_np = data.numpy()
    for i in range(data_np.shape[0]): # each row
        if data_np[i, 0] == index:
            return data_np[i, 1]

def add_predictions_to_qa_dict(data, predictions, mia):
    for e in data:
        question_id = int(e['question_id'])
        ans_pred = get_pred_ans(predictions, question_id)
        e['pred'] = mia[ans_pred]
    return data

config = io.read_config(jp(path_config, config_name + '.yaml'))
path_qa = jp(config['path_qa'], 'qa', 'testqa.json')
#read file with qa-pairs
if not os.path.exists(path_qa):
    raise FileNotFoundError
with open(path_qa, 'r') as f:
    qa_pairs_test = json.load(f)

# read answer to index map
path_map = jp(config['path_qa'], 'processed', 'map_answer_index.pickle')
if not os.path.exists(path_map):
    raise FileNotFoundError
with open(path_map, 'rb') as f:
    map_answer_index = pickle.load(f)
    map_index_answer = {v:k for k,v in map_answer_index.items()} # inver to have in inverted order

# load info about best epoch
path_logs = jp(config['logs_dir'], 'idrid_regions_single', config_name)
answers_test_best_epoch = torch.load(jp(path_logs, 'answers', 'answers_epoch_0.pt'))

# add model's prediction to qa pairs data
qa_pairs_test = add_predictions_to_qa_dict(qa_pairs_test, answers_test_best_epoch, map_index_answer)
# select questions for a specific question type
qa_pairs_test_grade = [e for e in qa_pairs_test if e['question_type']==question_type]
dicti_grade = {e['image_name'][:-4]:{'ans_model': e['pred'], 'ans_gt': e['answer']} for e in qa_pairs_test_grade}

# save dictionary
with open(jp(path_dict_grades, 'dict_' + question_type +'_'+config_name + '.pickle'), 'wb') as f:
    pickle.dump(dicti_grade, f, protocol=pickle.HIGHEST_PROTOCOL)