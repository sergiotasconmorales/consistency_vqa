# Project:
#   VQA
# Description:
#   Script for qualitatively checking images for which fovea questions were wrongly answered. 
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
from misc import dirs
import numpy as np
import shutil
from PIL import Image
from collections import Counter
from find_inconsistencies_dme_fovea_questions import add_predictions_to_qa_dict

subsi = 'test' # which subset to analyze. val or test. If test, inference answers (0) have to be available
output_folder = 'wrong_fovea' # folder to copy images for which fovea question was incorrectly answered

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)
    # get config file name
    config_file_name = args.path_config.split('/')[-1].split('.')[0]
    # path to logs
    path_logs = jp(config['logs_dir'], 'idrid_regions_single', config_file_name)

    path_output = jp(path_logs, output_folder)
    dirs.create_folder(path_output)

    path_images = jp(config['path_img'], subsi)

    path_best_info = jp(path_logs, 'best_checkpoint_info.pt')
    if not os.path.exists(path_best_info):
        raise FileNotFoundError

    # read test qa pairs
    path_qa = jp(config['path_qa'], 'qa', subsi + 'qa.json')
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
    info = torch.load(path_best_info)
    if subsi == 'val':
        answers_test_best_epoch = torch.load(jp(path_logs, 'answers', 'answers_epoch_' + str(info['epoch']) + '.pt'))
    elif subsi == 'test':
        answers_test_best_epoch = torch.load(jp(path_logs, 'answers', 'answers_epoch_0.pt'))


    qa_pairs_test = add_predictions_to_qa_dict(qa_pairs_test, answers_test_best_epoch, map_index_answer)

    # query wrongly answered fovea questions
    images_wrong_fovea_questions = [e['image_name'] for e in qa_pairs_test if e['question_type'] == 'fovea' and e['answer'] != e['pred']]

    answers_wrong_fovea_questions = [e['answer'] for e in qa_pairs_test if e['question_type'] == 'fovea' and e['answer'] != e['pred']]

    print('There are', len(images_wrong_fovea_questions), 'images for which fovea question was wrongly answered.')

    print(Counter(answers_wrong_fovea_questions).most_common())


    #copy images to output folder
    dirs.create_folders_within_folder(path_output, ['FP', 'FN'])
    for img in images_wrong_fovea_questions:
        pred = [e['pred'] for e in qa_pairs_test if e['question_type'] == 'fovea' and e['image_name'] == img][0]
        gt = [e['answer'] for e in qa_pairs_test if e['question_type'] == 'fovea' and e['image_name'] == img][0]
        if pred == 'yes' and gt == 'no':
            shutil.copy(jp(path_images, img), jp(path_output, 'FP', img))
        else:
            shutil.copy(jp(path_images, img), jp(path_output, 'FN', img))

    num_cols_rows = int(np.ceil(np.sqrt(len(images_wrong_fovea_questions))))

    fig, ax = plt.subplots(num_cols_rows,num_cols_rows,figsize=(10, 10))
    cnt = 0
    for i in range(num_cols_rows):
        for j in range(num_cols_rows):
            if cnt < len(images_wrong_fovea_questions):
                img = np.array(Image.open(jp(path_images, images_wrong_fovea_questions[cnt])))
                ax[i,j].imshow(img)
                ax[i,j].axis('off')
                ax[i,j].set_title(images_wrong_fovea_questions[cnt])
                cnt += 1

    plt.show()

if __name__ == '__main__':
    main()