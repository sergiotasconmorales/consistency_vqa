# Project:
#   VQA
# Description:
#   Script to analyze validation answers by question type
# Author: 
#   Sergio Tascon-Morales

import os 
from os.path import join as jp
import torch 
import json
import pickle
import misc.io as io
from matplotlib import pyplot as plt
import random
import numpy as np
from PIL import Image
from misc import general
from plot import plot_factory

samples_to_show = 10
desired_q_type = 'inside'
subsi = 'val' # which subset to analyze. val or test. If test, inference answers (0) have to be available

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def acc(pred, gt):
    return torch.eq(pred, gt).sum()/pred.shape[0]

def get_pred_ans(data, index):
    data_np = data.numpy()
    for i in range(data_np.shape[0]): # each row
        if data_np[i, 0] == index:
            return data_np[i, 1]


def compute_accuracies(config, config_file_name, subsi):
    # path to logs
    path_logs = jp(config['logs_dir'], config['dataset'], config_file_name)

    path_best_info = jp(path_logs, 'best_checkpoint_info.pt')
    if not os.path.exists(path_best_info):
        raise FileNotFoundError

    # read subset's qa pairs
    path_qa = jp(config['path_qa'], 'qa', subsi + 'qa.json')
    if not os.path.exists(path_qa):
        raise FileNotFoundError
    with open(path_qa, 'r') as f:
        qa_pairs_val = json.load(f)

    # read answer to index map
    path_map = jp(config['path_qa'], 'processed', 'map_answer_index.pickle')
    if not os.path.exists(path_map):
        raise FileNotFoundError
    with open(path_map, 'rb') as f:
        map_answer_index = pickle.load(f)
        map_index_answer = {v:k for k,v in map_answer_index.items()} # invert to have in inverted order

    # load info about best epoch
    info = torch.load(path_best_info)
    if subsi == 'val':
        answers_val_best_epoch = torch.load(jp(path_logs, 'answers', 'answers_epoch_' + str(info['epoch']) + '.pt'))
    elif subsi == 'test':
        answers_val_best_epoch = torch.load(jp(path_logs, 'answers', 'answers_epoch_0.pt'))

    # get gt and predicted answers for each question type
    dict_val_groups_pred, dict_val_groups_gt, q_types = general.group_answers_by_type(answers_val_best_epoch, qa_pairs_val, map_answer_index)

    # for each type, compute accuracy and print it
    total_unique_questions = 0
    total_right = 0
    mazamorra = torch.zeros_like(answers_val_best_epoch)
    accuracies = {}
    for (question_type, ans_pred), (_, ans_gt) in zip(dict_val_groups_pred.items(), dict_val_groups_gt.items()):
        assert ans_pred.shape[0] == ans_gt.shape[0]
        accuracies[question_type] = 100*acc(ans_pred, ans_gt).item()
        total_right += torch.eq(ans_pred, ans_gt).sum().item()
        total_unique_questions += ans_gt.shape[0]
        #print(question_type, "{:.2f}".format(100*acc(ans_pred, ans_gt).item()))

    accuracies['overall'] = 100*(total_right/total_unique_questions)
    #print('Overall: ', "{:.2f}".format(100*acc(mazamorra[:,0], mazamorra[:,1]).item()))

    return accuracies


def main():
    # read config file
    config = io.read_config(args.path_config)
    # get config file name
    config_file_name = args.path_config.split('/')[-1].split('.')[0]

    accuracies = compute_accuracies(config, config_file_name, subsi)

    print(accuracies)

if __name__ == '__main__':
    main()