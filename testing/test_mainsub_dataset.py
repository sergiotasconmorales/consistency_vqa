# Project:
#   VQA
# Description:
#   Script to check integrity of dataset that contains main and sub-questions after running modify_dme_main_questions_sub_questions.py
# Author: 
#   Sergio Tascon-Morales

import json
import pickle
import random 
from os.path import join as jp

path_ref = '/home/sergio814/Documents/PhD/code/data/dme_dataset_6_balanced_mainsub/qa'
path_mainsub = '/home/sergio814/Documents/PhD/code/data/dme_dataset_6_balanced_mainsub/processed'

subset_to_check = 'train'
num_samples_to_check = 1000

def find_sample_by_index(data, the_id):
    query = [e for e in data if e['question_id'] == the_id]
    assert len(query) == 1
    return query[0]

path_qa_ref = jp(path_ref, subset_to_check + 'qa.json')
path_qa_mainsub = jp(path_mainsub, subset_to_check + 'set.pickle')

# open ref qa pairs
with open(path_qa_ref, 'r') as f:
    ref = json.load(f)

with open(path_qa_mainsub, 'rb') as f:
    mainsub = pickle.load(f) 

# randomly choose a sample from mainsub and check that both main and sub questions refer to same image in ref, and that answers agree
for i in range(num_samples_to_check):
    sample = mainsub[random.randint(0, len(mainsub)-1)]
    print('Sample', i, '/', num_samples_to_check)
    ref_main = find_sample_by_index(ref, sample['main_question_id'])
    ref_sub = find_sample_by_index(ref, sample['sub_question_id'])
    print(">>Questions refer to same image:", ref_main['image_name'] == ref_sub['image_name'])
    print(">>Answer to main question is correct:", sample['main_answer'] == ref_main['answer'])
    print(">>Answer to sub question is correct:", sample['sub_answer'] == ref_sub['answer'])
