# Project:
#   VQA
# Description:
#   Script to create a consistency dataset with counterexamples. Every sample is two QA pairs and a binary label (consistent or inconsistent)
#   Only binary questions from VQA-Introspect are considered in this initial stage.
# Author: 
#   Sergio Tascon-Morales

import json
from os.path import join as jp
from tqdm import tqdm

path_introspect = '/home/sergio814/Documents/PhD/code/data/VQAIntrospect'
path_output = '/home/sergio814/Documents/PhD/code/data/consistency_data'
base_name = 'VQAIntrospect_<>v1.0.json'
subsets = ['train', 'val']

def change_binary_answer(answer):
    if answer in ['yes', 'yea', 'yeah']:
        return 'no'
    elif answer in ['no', 'nope', 'naa']:
        return 'yes'
    else:
        raise ValueError('Unknown answer')

def is_binary(answer):
    if answer in ['yes', 'no', 'yeah', 'nope', 'yea']:
        return True
    else:
        return False




# for each subset
for s in subsets:
    samples = [] # list for data for current subset
    # read json
    with open(jp(path_introspect, base_name.replace('<>', s))) as f:
        data = json.load(f)
    print('Processing', s, 'samples...')
    for k, v in tqdm(data.items()):
        # k is the question id
        # v is a dictionary with all info
        mainq = v['reasoning_question']
        maina = v['reasoning_answer_most_common']
        for intr in v['introspect']: # for each sub-question
            # check if it's binary. If so, generate two samples
            if len(intr['sub_qa'])==1:
                isbin = is_binary(intr['sub_qa'][0]['sub_answer'])
                if isbin:
                    # if it's binary, add it to 
                    subq = intr['sub_qa'][0]['sub_question']
                    suba = intr['sub_qa'][0]['sub_answer']
                    samples.append({'main_question': mainq, 'main_answer': maina, 'sub_question': subq, 'sub_answer': suba, 'label': 1})
                    samples.append({'main_question': mainq, 'main_answer': maina, 'sub_question': subq, 'sub_answer': change_binary_answer(suba), 'label': 0})
            elif len(intr['sub_qa'])>1:
                for elem in intr['sub_qa']:
                    isbin = is_binary(elem['sub_answer'])
                    if isbin:
                        subq = elem['sub_question']
                        suba = elem['sub_answer']
                        samples.append({'main_question': mainq, 'main_answer': maina, 'sub_question': subq, 'sub_answer': suba, 'label': 1})
                        samples.append({'main_question': mainq, 'main_answer': maina, 'sub_question': subq, 'sub_answer': change_binary_answer(suba), 'label': 0})
    
    # save data for current subset
    with open(jp(path_output, s + '.json'), 'w') as f:
        json.dump(samples, f)


    
