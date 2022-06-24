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
config_baseline = '947'
config_squint = '973'
config_mine = '962'
path_test_qa = '/home/sergio814/Documents/PhD/code/data/dme_dataset_10_balanced/processed/testset.pickle'

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

cnt = 0
cnt_mine = 0
cnt_squint = 0
cnt_baseline = 0
cnt_total = 0

normal_right = {'baseline': 0, 'squint': 0, 'mine': 0}
cf_right = {'baseline': 0, 'squint': 0, 'mine': 0}

# for each index, check if question is main
for i in tqdm(range(0, pred_baseline.shape[0])):
    q_id = pred_baseline[i,0].item()
    if di_qid[q_id]['role'] == 'main' and '_.jpg' in di_qid[q_id]['image_name']: # counterfacutal samples
        cnt+=1
        # check answers
        ans_gt = di_qid[q_id]['answer_index']
        ans_baseline = pred_baseline[i,1].item()
        ans_squint = pred_squint[i,1].item()
        ans_mine = pred_mine[i,1].item()
        all_same_ = are_all_same(ans_gt, ans_baseline, ans_squint, ans_mine)

        all_same__mine = are_all_same(ans_gt, ans_mine) 
        if all_same__mine:
            cf_right['mine'] += 1
        all_same__squint = are_all_same(ans_gt, ans_squint) 
        if all_same__squint:
            cf_right['squint'] += 1
        all_same__baseline = are_all_same(ans_gt, ans_baseline) 
        if all_same__baseline:
            cf_right['baseline'] += 1


        # now answer for counterfactual
        ans_id_main_cf = [(e['answer_index'], e['question_id']) for e in test_questions if e['role'] == 'main' and e['image_name'] == di_qid[q_id]['image_name'].replace('_.', '.')]
        if not ans_id_main_cf:
            continue # current image does not have a counterfactual
        else:
            cnt_total += 1
            # counterfactual is there
            ans_gt_cf = ans_id_main_cf[0][0] #only one main question can exist
            ans_baseline_cf = pred_baseline[torch.where(pred_baseline[:,0]==ans_id_main_cf[0][1])[0].item(), 1].item()
            ans_squint_cf = pred_squint[torch.where(pred_squint[:,0]==ans_id_main_cf[0][1])[0].item(), 1].item()
            ans_mine_cf = pred_mine[torch.where(pred_mine[:,0]==ans_id_main_cf[0][1])[0].item(), 1].item()
            all_same =are_all_same(ans_gt_cf, ans_baseline_cf, ans_squint_cf, ans_mine_cf)

            all_same_mine = are_all_same(ans_gt_cf, ans_mine_cf)
            if all_same_mine:
                normal_right['mine'] += 1
            all_same_squint = are_all_same(ans_gt_cf, ans_squint_cf)
            if all_same_squint:
                normal_right['squint'] += 1
            all_same_baseline = are_all_same(ans_gt_cf, ans_baseline_cf)
            if all_same_baseline:
                normal_right['baseline'] += 1


            if all_same__mine and all_same_mine:
                cnt_mine += 1

            if all_same__squint and all_same_squint:
                cnt_squint += 1

            if all_same__baseline and all_same_baseline:
                cnt_baseline += 1

            if all_same and all_same_: # if all models correctly changed the answer
                

                print('For images', di_qid[q_id]['image_name'], 'and', di_qid[q_id]['image_name'].replace('_.', '.'), 'all three models changed answer correctly')

                # Print sub-questions for _
                curr_image =  di_qid[q_id]['image_name']
                ids_sub = [e['question_id'] for e in test_questions if e['image_name'] == curr_image and e['role'] == 'sub']
                print('Main:', q_id, di_qid[q_id]['question'], 'about', di_qid[q_id]['image_name'], 'Ans:', di_qid[q_id]['answer'])
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

                # Print sub-questions for orig
                curr_image =  di_qid[q_id]['image_name'].replace('_.', '.')
                ids_sub = [e['question_id'] for e in test_questions if e['image_name'] == curr_image and e['role'] == 'sub']

                print('Main:', ans_id_main_cf[0][1], di_qid[ans_id_main_cf[0][1]]['question'], 'about', di_qid[ans_id_main_cf[0][1]]['image_name'], 'Ans:', di_qid[ans_id_main_cf[0][1]]['answer'])
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


print('mine normal right:', normal_right['mine'], '/', cnt)
print('mine cf right:', cf_right['mine'], '/', cnt)
print('squint normal right:', normal_right['squint'], '/', cnt)
print('squint cf right:', cf_right['squint'], '/', cnt)
print('baseline normal right:', normal_right['baseline'], '/', cnt)
print('baseline cf right:', cf_right['baseline'], '/', cnt)

a = 42

#print('Total all models right:', cnt)
print('Baseline:', cnt_baseline, '/', cnt_total)
print('SQuINT:', cnt_squint, '/', cnt_total)
print('Mine:', cnt_mine, '/', cnt_total)



"""

    a = 42

    ans_gt = di_qid[q_id]['answer_index']
    ans_baseline = pred_baseline[i,1].item()
    ans_squint = pred_squint[i,1].item()
    ans_mine = pred_mine[i,1].item()
    all_same = are_all_same(ans_gt, ans_baseline, ans_squint, ans_mine)
    if all_same:
        
        # now find all sub-questions that refer to this image and cound, for every model, how many of them are right
        curr_image =  di_qid[q_id]['image_name']
        ids_sub = [e['question_id'] for e in test_questions if e['image_name'] == curr_image and e['role'] == 'sub']
        if len(ids_sub)>2:
            print('Main:', q_id, di_qid[q_id]['question'], 'about', di_qid[q_id]['image_name'], 'Ans:', di_qid[q_id]['answer'])
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
"""
