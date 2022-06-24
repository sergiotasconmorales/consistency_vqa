# Project:
#   VQA
# Description:
#   Script to find qualitative examples for MICCAI paper 2022
# Author: 
#   Sergio Tascon-Morales

import pickle
import torch
from os.path import join as jp
from tqdm import tqdm


path_logs = '/home/sergio814/Documents/PhD/code/logs/idrid_regions_single'
config_baseline = '684'
config_squint = '940'
config_mine = '729'
path_test_qa = '/home/sergio814/Documents/PhD/code/data/dme_dataset_8_balanced/processed/testset.pickle'

def are_all_same(ans_gt, *args):
    result = True
    for ans in args:
        if ans != ans_gt:
            result = False
    return result

# I have to find examples for which all 3 models got the main question correctly, but baseline and squint messed the sub-questions.

with open(path_test_qa, 'rb') as f:
    test_questions = pickle.load(f)

# generate dictionary with question ids as keys
di_qid = {e['question_id']:e for e in test_questions}

# read predictions
pred_baseline = torch.load(jp(path_logs, 'config_' + config_baseline, 'answers', 'answers_epoch_0.pt'))
pred_squint = torch.load(jp(path_logs, 'config_' + config_squint, 'answers', 'answers_epoch_0.pt'))
pred_mine = torch.load(jp(path_logs, 'config_' + config_mine, 'answers', 'answers_epoch_0.pt'))

# for each index, check if question is main
for i in tqdm(range(0, pred_baseline.shape[0])):
    q_id = pred_baseline[i,0].item()
    if di_qid[q_id]['role'] == 'main':
        ans_gt = di_qid[q_id]['answer_index']
        ans_baseline = pred_baseline[i,1].item()
        ans_squint = pred_squint[i,1].item()
        ans_mine = pred_mine[i,1].item()
        all_same = are_all_same(ans_gt, ans_mine) #* SQuINT and baseline excluded
        if all_same:
            
            # now find all sub-questions that refer to this image and cound, for every model, how many of them are right
            curr_image =  di_qid[q_id]['image_name']
            ids_sub = [e['question_id'] for e in test_questions if e['image_name'] == curr_image and e['role'] == 'sub']
            if len(ids_sub)==2:
                print('Main:', q_id, di_qid[q_id]['question'], 'about', di_qid[q_id]['image_name'], 'Ans:', di_qid[q_id]['answer'], 'encoded_ans baseline:', ans_baseline, 'enc_ans_squint:', ans_squint, 'enc_ans_mine:', ans_mine)
                number_sub_right = {'baseline':0, 'squint':0, 'mine':0}
                for sub_id in ids_sub:
                    row = torch.where(pred_baseline[:,0]==sub_id)[0].item()
                    ans_sub_gt = di_qid[sub_id]['answer_index']
                    number_sub_right['baseline'] += int(are_all_same(ans_sub_gt, pred_baseline[row,1].item()))
                    number_sub_right['squint'] += int(are_all_same(ans_sub_gt, pred_squint[row,1].item()))
                    number_sub_right['mine'] += int(are_all_same(ans_sub_gt, pred_mine[row,1].item()))
                    print('     Sub:', di_qid[sub_id]['question'], 'gt:', ans_sub_gt, ', mask:', di_qid[sub_id]['mask_name'], 'ans baseline:', pred_baseline[row,1].item(), 'ans_squint:', pred_squint[row,1].item(), 'ans_mine:', pred_mine[row,1].item())
                if number_sub_right['mine'] > number_sub_right['squint'] or number_sub_right['mine'] > number_sub_right['baseline']:
                    print('Good example found:', q_id)
            else: 
                continue
