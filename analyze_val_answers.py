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

torch.manual_seed(1234) # use same seed for reproducibility

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

    """
    # now visualize some samples
    cnt = 0 # count error samples
    sample_type = 'pajarito'
    while cnt < samples_to_show:
        while sample_type != desired_q_type:
            sample = random.choice(qa_pairs_val) # here I have the question, the answer, the image names, the question type and the question id
            sample_type = sample['question_type']
        sample_index = int(sample['question_id'])
        pred_ans_index = get_pred_ans(answers_val_best_epoch, sample_index)
        pred_ans_text = map_index_answer[pred_ans_index]
        if sample['answer'] == pred_ans_text:
            sample_type = 'pajarito'
            continue  # looking for errors
        else: # show only errors
            img = np.array(Image.open(jp(path_images, sample['image_name'])))

            # depending on question type, show something different
            if sample['question_type'] == 'inside':
                # open mask
                mask = np.array(Image.open(jp(path_masks, 'maskA', sample['mask_name'])))
                print('Question:', sample['question'])
                print('id:', sample['question_id'], 'image name:', sample['image_name'])
                print('Ans GT: ' + sample['answer'] + ', ans pred: ' + pred_ans_text)
                plot_factory.overlay_mask(img, mask, mask, alpha=0.3)
            elif sample['question_type'] == 'whole':
                plt.imshow(img)
                print('Question:', sample['question'])
                print('id:', sample['question_id'], 'image name:', sample['image_name'])
                print('Ans GT: ' + sample['answer'] + ', ans pred: ' + pred_ans_text)
            elif sample['question_type'] == 'grade':
                # Here simply print
                print('Question:', sample['question'])
                print('id:', sample['question_id'], 'image name:', sample['image_name'])
                print('Ans GT: ' + str(sample['answer']) + ', ans pred: ' + str(pred_ans_text))

            sample_type = 'pajarito'
            #f, (ax1, ax2) = plt.subplots(1, 2)
            #st = f.suptitle(sample['question'] + '\n' + 'Ans GT: ' + sample['answer'][0] + ', ans pred: ' + pred_ans_text)
            #ax1.imshow(img1)
            #ax2.imshow(img2)
            #ax1.xaxis.set_visible(False)
            #ax1.yaxis.set_visible(False)
            #ax2.xaxis.set_visible(False)
            #ax2.yaxis.set_visible(False)
            #cnt += 1
    """
if __name__ == '__main__':
    main()