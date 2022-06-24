# Project:
#   VQA
# Description:
#   ...
# Author: 
#   Sergio Tascon-Morales


import os 
from os.path import join as jp
import torch 
import json
import pickle
import misc.io as io
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import numpy as np
from PIL import Image
from misc import general
from plot import plot_factory


subsi = 'val'

args = io.get_config_file_name()

def acc(pred, gt):
    return torch.eq(pred, gt).sum()/pred.shape[0]

def create_pred_gt_array(preds, gt):
    # preds is a tensor with question id in first column and predicted answer (encoded) in the second colum
    # gt is the info from pickle file
    big_array = torch.zeros_like(preds)
    print('Creating array of preds and gt...')
    for row in tqdm(range(preds.shape[0])):
        q_id = preds[row, 0].item()
        ans_pred = preds[row, 1].item()
        ans_gt = gt[q_id]
        big_array[row, 0] = ans_pred
        big_array[row, 1] = ans_gt
    return big_array


def compute_accuracies(config, config_file_name, subsi):
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

    # get gt and predicted answers for each question type
    array_pred = create_pred_gt_array(answers_val_best_epoch, {e['question_id']: e['answer_index'] for e in qa_pairs_val})

    accuracy = 100*acc(array_pred[:,0], array_pred[:,1]).item()

    return accuracy


def main():
    # read config file
    config = io.read_config(args.path_config)
    # get config file name
    config_file_name = args.path_config.split('/')[-1].split('.')[0]

    accuracy = compute_accuracies(config, config_file_name, subsi)

    print(accuracy)

if __name__ == '__main__':
    main()