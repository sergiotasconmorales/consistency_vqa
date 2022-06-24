# Project:
#   VQA
# Description:
#   Script to find the answer imbalance in the training dataset for each image, and for every question class. This is for the single region dataset
# Author: 
#   Sergio Tascon-Morales

import os
import pickle
from os.path import join as jp
import pandas as pd
import torch

subset = 'test'
epoch = '0'
path_answers_model = '/home/sergio814/Documents/PhD/code/logs/idrid_regions_single/config_013/answers/answers_epoch_' + epoch + '.pt'

path_images = '/home/sergio814/Documents/PhD/code/data/idrid_ODEX_balanced/visual'
path_data = '/home/sergio814/Documents/PhD/code/data/idrid_ODEX_balanced/processed'
dataset_name = subset + 'set.pickle'

path_results = '/home/sergio814/Documents/PhD/code/logs/idrid_regions_single/config_013/'

abnormalities = {'optic': 'OD', 'hemorrhages': 'HE', 'microaneurysms': 'MA', 'soft': 'SE', 'hard': 'EX'}

# list images of subset
images_list = os.listdir(jp(path_images, subset))

counts_gt = {img:{suffix: [0,0] for suffix in abnormalities.values()} for img in images_list}
counts_model = {img:{suffix: [0,0] for suffix in abnormalities.values()} for img in images_list}

# read pre-processed data
with open(jp(path_data, dataset_name), 'rb') as f:
    data = pickle.load(f)

# read model's answers
answers = torch.load(path_answers_model)['results']

for elem, answer in zip(data, answers):
    if elem['answer'] == 'yes':
        counts_gt[elem['image_name']][abnormalities[elem['question_tokens'][2]]][0] += 1
    elif elem['answer'] == 'no':
        counts_gt[elem['image_name']][abnormalities[elem['question_tokens'][2]]][1] += 1
    if answer[1] < 1:
        counts_model[elem['image_name']][abnormalities[elem['question_tokens'][2]]][0] += 1
    else:
        counts_model[elem['image_name']][abnormalities[elem['question_tokens'][2]]][1] += 1

df_gt = pd.DataFrame(columns= ['Image', 'OD_yes', 'OD_no', 'HE_yes', 'HE_no', 'MA_yes', 'MA_no', 'SE_yes', 'SE_no', 'EX_yes', 'EX_no'])
df_model = pd.DataFrame(columns= ['Image', 'OD_yes', 'OD_no', 'HE_yes', 'HE_no', 'MA_yes', 'MA_no', 'SE_yes', 'SE_no', 'EX_yes', 'EX_no'])
cnt = 0
for elem_gt, elem_m in zip(counts_gt.items(), counts_model.items()):
    entry_gt = [elem_gt[0], elem_gt[1]['OD'][0], elem_gt[1]['OD'][1], elem_gt[1]['HE'][0], elem_gt[1]['HE'][1], elem_gt[1]['MA'][0], elem_gt[1]['MA'][1], elem_gt[1]['SE'][0], elem_gt[1]['SE'][1], elem_gt[1]['EX'][0], elem_gt[1]['EX'][1]]
    entry_m = [elem_m[0], elem_m[1]['OD'][0], elem_m[1]['OD'][1], elem_m[1]['HE'][0], elem_m[1]['HE'][1], elem_m[1]['MA'][0], elem_m[1]['MA'][1], elem_m[1]['SE'][0], elem_m[1]['SE'][1], elem_m[1]['EX'][0], elem_m[1]['EX'][1]]
    df_gt.loc[cnt] = entry_gt
    df_model.loc[cnt] = entry_m
    cnt += 1
a = 42
df_gt.to_csv(jp(path_results, subset + '_gt_class_imbalance_per_image_and_question_epoch' + str(epoch) + '.csv'))
df_model.to_csv(jp(path_results, subset + '_model_class_imbalance_per_image_and_question_epoch' + str(epoch) + '.csv'))
print(counts_gt)