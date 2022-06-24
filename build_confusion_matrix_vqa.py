# Project:
#   CompVQA
# Description:
#   Script for computing and displaying the confusion matrices for the dme dataset
# Author: 
#   Sergio Tascon-Morales


from os.path import join as jp
import misc.io as io
import torch
import os
import pickle
import pandas as pd
import numpy as np
import json
from misc import general
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics as skmetrics
from plot import plot_confusion_matrix as pcm

torch.manual_seed(1234) # use same seed for reproducibility

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def find_entry(data, q_id):
    for e in data:
        if int(e['question_id']) == q_id:
            return e

def get_ordered_labels(data, ids):
    answers_gt = []
    for i in tqdm(ids):
        answers_gt.append(find_entry(data, i)['answer_index'])
    return np.array(answers_gt, dtype = np.int64)

def main():
    # read config file
    config = io.read_config(args.path_config) 

    config_file_name = args.path_config.split("/")[-1].split(".")[0]

    path_logs = jp(config['logs_dir'], config['dataset'], config_file_name)


    # first, generate plots for best validation epoch
    best_epoch_info_path = jp(path_logs, 'best_checkpoint_info.pt')
    best_epoch_info = torch.load(best_epoch_info_path, map_location=torch.device('cpu'))

    best_epoch_index = best_epoch_info['epoch']
    # now go to answers folder and read info from there
    path_val_answers_file = jp(path_logs, 'answers', 'answers_epoch_' + str(best_epoch_index) + '.pt')
    answers_best_val_epoch = torch.load(path_val_answers_file, map_location=torch.device('cpu')) # dictionary with keys: results, answers. results contains tensor with (question_index, model's answer), answers is  (target, prob)

    ans = answers_best_val_epoch[:,1].numpy() # all answers predicted by the model

    # Now I need to gather the answers from the gt in the same order of indexes present in answers_best_val_epoch
    path_gt = jp(config['path_qa'], 'processed', 'valset.pickle')
    if os.path.exists(path_gt):
        with open(path_gt, 'rb') as f:
            gt_data = pickle.load(f)

    # open non-processed qa pairs for dividing the samples by question type later
    path_gt_raw = jp(config['path_qa'], 'qa', 'valqa.json')
    if os.path.exists(path_gt_raw):
        with open(path_gt_raw, 'rb') as f:
            qa_pairs_val = json.load(f)

    # read answer to index map
    path_map = jp(config['path_qa'], 'processed', 'map_answer_index.pickle')
    if not os.path.exists(path_map):
        raise FileNotFoundError
    with open(path_map, 'rb') as f:
        map_answer_index = pickle.load(f)
        map_index_answer = {v:k for k,v in map_answer_index.items()} # inver to have in inverted order

    print('Obtaining GT labels...')
    gt = get_ordered_labels(gt_data, list(answers_best_val_epoch[:,0].numpy()))

    # now plot confusion matrix
    lbls = list(set(list(gt)))
    cm = skmetrics.confusion_matrix(gt, ans, labels=lbls)

    df_cm = pd.DataFrame(cm, index=lbls, columns=lbls)

    pcm.pretty_plot_confusion_matrix(df_cm, save_img=True, save_path = path_logs, file_name = config_file_name)

    # now generate confusion matrix for each question type
    dict_val_groups_pred, dict_val_groups_gt, q_types = general.group_answers_by_type(answers_best_val_epoch, qa_pairs_val, map_answer_index)

    # now for each group, plot and save confusion matrix
    for (question_type, ans_pred), (_, ans_gt) in zip(dict_val_groups_pred.items(), dict_val_groups_gt.items()):
        cm = skmetrics.confusion_matrix(ans_gt, ans_pred, labels=list(set(ans_gt.numpy())))
        df_cm = pd.DataFrame(cm, index=list(set(ans_gt.numpy())), columns=list(set(ans_gt.numpy())))
        pcm.pretty_plot_confusion_matrix(df_cm, save_img=True, save_path = path_logs, file_name = config_file_name + question_type)

if __name__ == '__main__':
    main()