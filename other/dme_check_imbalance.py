# Project:
#   VQA
# Description:
#   Script to check the imbalance in the DME VQA dataset
# Author: 
#   Sergio Tascon-Morales

import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
from os.path import join as jp
from collections import Counter

from torch._C import Value

path_gt = '/home/sergio814/Documents/PhD/code/data/dme_dataset_6_balanced/processed'
path_map = jp(path_gt, 'map_answer_index.pickle')
subsets = ['train', 'val', 'test']

path_ans = '/home/sergio814/Documents/PhD/code/logs/idrid_regions_single/config_076/answers'

def get_pred_ans(data, index):
    data_np = data.numpy()
    for i in range(data_np.shape[0]): # each row
        if data_np[i, 0] == index:
            return data_np[i, 1]

def add_predictions_to_qa_dict(data, predictions, mia):
    print('Adding predicted answers to main data list...')
    for e in tqdm(data):
        question_id = int(e['question_id'])
        ans_pred = get_pred_ans(predictions, question_id)
        e['pred'] = mia[ans_pred]
    return data

for s in subsets:
    print(50*'*')
    print("Subset: ", s)
    with open(jp(path_gt, s +'set.pickle'), 'rb') as f:
        data = pickle.load(f)

    if not os.path.exists(path_map):
        raise FileNotFoundError
    with open(path_map, 'rb') as f:
        map_answer_index = pickle.load(f)
        map_index_answer = {v:k for k,v in map_answer_index.items()} # inver to have in inverted order

    # read model's answers
    if s== 'train':
        answers_best_epoch = torch.load(jp(path_ans, 'answers_epoch_1000.pt'))
    elif s == 'val':
        answers_best_epoch = torch.load(jp(path_ans, 'answers_epoch_2000.pt'))
    elif s == 'test':
        answers_best_epoch = torch.load(jp(path_ans, 'answers_epoch_0.pt'))
    else:
        raise ValueError

    data = add_predictions_to_qa_dict(data, answers_best_epoch, map_index_answer)

    total_samples = len(data)
    # now separate in groups by question type
    samples_inside = [e for e in data if e['question_type'] == 'inside']
    samples_whole = [e for e in data if e['question_type'] == 'whole']
    samples_grade = [e for e in data if e['question_type'] == 'grade']
    samples_fovea = [e for e in data if e['question_type'] == 'fovea']

    ans_inside_gt = [e['answer'] for e in samples_inside]
    ans_whole_gt = [e['answer'] for e in samples_whole]
    ans_grade_gt = [e['answer'] for e in samples_grade]
    ans_fovea_gt = [e['answer'] for e in samples_fovea]

    ans_inside_pred = [e['pred'] for e in samples_inside]
    ans_whole_pred = [e['pred'] for e in samples_whole]
    ans_grade_pred = [e['pred'] for e in samples_grade]
    ans_fovea_pred = [e['pred'] for e in samples_fovea]

    print('IMBALANCES FOR GT')

    print('Inside')
    print('Total:', len(samples_inside))
    print('Distr:', Counter(ans_inside_gt).most_common())

    print('Whole')
    print('Total:', len(samples_whole))
    print('Distr:', Counter(ans_whole_gt).most_common())

    print('Grade')
    print('Total:', len(samples_grade))
    print('Distr:', Counter(ans_grade_gt).most_common())

    print('Fovea')
    print('Total:', len(samples_fovea))
    print('Distr:', Counter(ans_fovea_gt).most_common())

    print('TOTAL SAMPLES:', total_samples)

    # now analize what the model said
    print('IMBALANCES FOR PREDICTIONS')

    print('Inside')
    print('Total:', len(samples_inside))
    print('Distr:', Counter(ans_inside_pred).most_common())

    print('Whole')
    print('Total:', len(samples_whole))
    print('Distr:', Counter(ans_whole_pred).most_common())

    print('Grade')
    print('Total:', len(samples_grade))
    print('Distr:', Counter(ans_grade_pred).most_common())

    print('Fovea')
    print('Total:', len(samples_fovea))
    print('Distr:', Counter(ans_fovea_pred).most_common())
