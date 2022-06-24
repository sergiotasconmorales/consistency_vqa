# Project:
#   VQA
# Description:
#   Script to copmute the accuracy and consistency metrics for the introspect dataset.
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
from tqdm import tqdm
import numpy as np
from PIL import Image
from misc.printer import print_event as pe
from plot import plot_factory
from metrics import metrics
from compute_consistency import add_predictions_to_qa_dict

subsi = 'val'

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def main():

    # read config name from CLI argument --path_config
    args = io.get_config_file_name()

    # read config file
    config = io.read_config(args.path_config)
    # get config file name
    config_file_name = args.path_config.split('/')[-1].split('.')[0]

    # path to logs
    path_logs = jp(config['logs_dir'], config['dataset'], config_file_name)
    # path to images
    path_images = jp(config['path_img'], subsi)
    #path_masks = jp(config['path_masks'], subsi)

    path_best_info = jp(path_logs, 'best_checkpoint_info.pt')
    if not os.path.exists(path_best_info):
        raise FileNotFoundError

    # read subset's qa pairs
    path_qa = jp(config['path_qa'], 'processed', subsi + 'set.pickle')
    if not os.path.exists(path_qa):
        raise FileNotFoundError
    with open(path_qa, 'rb') as f:
        qa_pairs_val = pickle.load(f)

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

    # for convenience, build dictionary question_id -> answer_index from answers_val_best_epoch
    q_id_ans = {answers_val_best_epoch[i,0].item():answers_val_best_epoch[i,1].item() for i in range(answers_val_best_epoch.shape[0])}

    # important: answers_val_best_epoch has repeated ids because main questions appear several times. This is important when computing the accuracy.
    #* step 1: add predictions to qa_pairs_val
    pe("Adding predictions to list of dicts")
    for e in tqdm(qa_pairs_val):
        e['pred'] = q_id_ans[e['question_id']]

    #* step 2: build array (pred, gt) from qa_pairs_val

    pe("Building array with predictions and GT values")
    pred_gt = torch.zeros_like(answers_val_best_epoch, dtype=torch.int64)
    for i, e in enumerate(tqdm(qa_pairs_val)):
        pred_gt[i, 0] = e['pred']
        pred_gt[i, 1] = e['answer_index']

    #* step 3: compute accuracy
    acc = torch.eq(pred_gt[:, 0], pred_gt[:, 1]).sum()/pred_gt.shape[0]

    #* step 4: compute consistency
    pivot = len([e for e in qa_pairs_val if e['role'] == 'main']) + len([e for e in qa_pairs_val if e['role'] == 'ind'])
    consistency = metrics.consistency_introspect(qa_pairs_val, pivot=pivot)
    
    print("Accuracy:", acc)
    print("Consistency:", consistency)

if __name__ == '__main__':
    main()