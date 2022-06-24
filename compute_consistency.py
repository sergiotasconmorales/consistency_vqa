# Project:
#   VQA
# Description:
#   Script to compute consistency after having performed inference with a model 
# Author: 
#   Sergio Tascon-Morales

import os 
from os.path import join as jp
import torch 
import json
import pickle
import misc.io as io
from matplotlib import pyplot as plt
from misc import printer
import numpy as np
from PIL import Image
from tqdm import tqdm
from metrics.metrics import consistency, consistencies_q2_q3
from collections import Counter
from misc import general, dirs
from plot import plot_factory as pf

torch.manual_seed(1234) # use same seed for reproducibility

def get_pred_ans(data, index):
    data_np = data.numpy()
    for i in range(data_np.shape[0]): # each row
        if data_np[i, 0] == index:
            return data_np[i, 1]

def add_predictions_to_qa_dict(data, predictions, mia):
    print("Adding precitions to qa dict ...")
    for e in tqdm(data):
        question_id = int(e['question_id'])
        ans_pred = get_pred_ans(predictions, question_id)
        e['pred'] = mia[ans_pred]
    return data

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def compute_consistency(config, config_file_name, q3_too=False):

    path_logs = jp(config['logs_dir'], 'idrid_regions_single', config_file_name)
    path_inconsistencies = jp(path_logs, 'inconsistencies')
    dirs.create_folder(path_inconsistencies)
    dirs.create_folders_within_folder(path_inconsistencies, ['zero', 'one', 'two'])

    # path to images and to masks
    path_images = jp(config['path_img'], 'test')
    path_masks = jp(config['path_masks'], 'test', 'maskA')

    path_best_info = jp(path_logs, 'best_checkpoint_info.pt')
    if not os.path.exists(path_best_info):
        raise FileNotFoundError

    # read validation qa pairs
    path_qa = jp(config['path_qa'], 'qa', 'testqa.json')
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
    answers_test_best_epoch = torch.load(jp(path_logs, 'answers', 'answers_epoch_0.pt'))

    qa_pairs_test = add_predictions_to_qa_dict(qa_pairs_test, answers_test_best_epoch, map_index_answer)

    if q3_too:
        c = consistencies_q2_q3(qa_pairs_test)
        return c
    else:
        c = consistency(qa_pairs_test)
        return c*100

    


def main():
    # read config file
    config = io.read_config(args.path_config)
    # get config file name
    config_file_name = args.path_config.split('/')[-1].split('.')[0]
    # path to logs

    c = compute_consistency(config, config_file_name)
    print('Consistency:', "{:.2f} %".format(c))


if __name__ == '__main__':
    main()