# Project:
#   VQA
# Description:
#   Script to test integrity of DME dataset, including unicity of QA ids, unicity of masks and balance
# Author: 
#   Sergio Tascon-Morales

import os
import json
from collections import Counter
from os.path import join as jp

from cv2 import QT_CHECKBOX

PATH_JSONS = '/home/sergio814/Documents/PhD/code/data/dme_dataset_8_balanced/qa'
PATH_MASKS = '/home/sergio814/Documents/PhD/code/data/dme_dataset_8_balanced/masks'

def status(b):
    # convert boolean to string
    if b:
        return 'Okay'
    else:
        return 'Failed'

def print_line():
    print(50*'-')

def print_in_order(di, ord, tot):
    # function to print elements of a dict that are in ord (ordered list) and show percentages of tot
    for e in ord:
        if e in di:
            print('Answer <' + str(e) + '>', ':', di[e], '/', tot, '(', '{:.2f}%'.format(100*di[e]/tot), ')')
    return

for s in ['train', 'val', 'test']:
    print_line()
    print('SUBSET: ', s.upper())
    with open(jp(PATH_JSONS, s+'qa.json'), 'r') as f:
        qa = json.load(f)


        print_line()
        print('UNICITY')
        # check unicity of ids
        unicity_id = len(qa) == len(set([e['question_id'] for e in qa]))
        print('Unicity of question ids:', status(unicity_id))

        # check unicity of masks
        num_mask_files = len(os.listdir(jp(PATH_MASKS, s, 'maskA'))) - 1 # subtracting whole_image mask
        num_mask_data = len(set([e['mask_name'] for e in qa if e['question_type'] == 'inside'])) 
        print('Unicity of masks:', status(num_mask_files==num_mask_data))
        print_line()
        # Analyze balance in the data
        # first, check how many of non-grade questions are 'sub'
        print('AMOUNT OF REAL SUB-QUESTIONS')
        sub_ind = len([e for e in qa if e['question_type'] != 'grade'])
        sub = len([e for e in qa if e['role'] == 'sub'])
        print('Amount of sub questions is', sub, 'out of', sub_ind, 'non-main questions(', '{:.2f}%'.format(100*sub/sub_ind), ')')
        print_line()

        # check total imbalance for answers
        print('GENERAL IMBALANCE')
        grade_answers = [e['answer'] for e in qa if e['question_type'] == 'grade']
        grade_answers_unique = list(set(grade_answers))
        grade_answers_unique.sort()
        nongrade_answers = [e['answer'] for e in qa if e['question_type'] != 'grade']
        nongrade_answers_unique = list(set(nongrade_answers))
        nongrade_answers_unique.sort()

        counts_grade_answers = dict(Counter(grade_answers))
        print('Grade questions')
        print_in_order(counts_grade_answers, grade_answers_unique, len(grade_answers))
        counts_nongrade_answers = dict(Counter(nongrade_answers))
        print('All other questions')
        print_in_order(counts_nongrade_answers, nongrade_answers_unique, len(nongrade_answers))
        print_line()

        print('PER-TYPE IMBALANCE')
        # next, check per type imbalances
        q_types = list(set([e['question_type'] for e in qa]))
        for qt in q_types:
            if qt == 'grade':
                continue # no need, already done
            qt_answers = [e['answer'] for e in qa if e['question_type'] == qt]
            qt_answers_unique = list(set(qt_answers))
            qt_answers_unique.sort()
            counts = dict(Counter(qt_answers))
            print(qt, 'questions')
            print_in_order(counts, qt_answers_unique, len(qt_answers))
        print_line()

        print('IMBALANCE FOR SUB-QUESTIONS CENTERED AT MACULA')
        sub_answers = [e['answer'] for e in qa if e['role'] == 'sub' and e['center'] == 'macula']
        sub_answers_unique = list(set(sub_answers))
        sub_answers_unique.sort()
        c = dict(Counter(sub_answers))
        print_in_order(c, sub_answers_unique, len(sub_answers))
        print_line()