# Project:
#   VQA
# Description:
#   Script to create VQA2loc dataset, which is the fusion between VQA2 and COCO-Regions
# Author: 
#   Sergio Tascon-Morales

from add_vqa_introspect_to_vqa2 import get_valid_question_id
from misc import dirs, io
from os.path import join as jp
import json
from tqdm import tqdm

# define paths
path_vqa2_qa = 'data/VQA2/qa'
path_coco_regions = '/home/sergio814/Documents/PhD/code/data/coco/coco_regions_3_balanced/qa'

path_output = 'data/VQA2loc2'
path_output_qa = jp(path_output, 'qa')

# create folder for output
dirs.create_folder(path_output)
dirs.create_folder(path_output_qa)


def get_image_id(image_name):
    num = image_name.split('_')[-1].split('.')[0]
    return int(num)

def create_list_of_repeated_answers(answer):
    # Creates list of 10 repeated answers so that they can be added to VQA2
    return [{'answer': answer, 'answer_confidence': 'yes', 'answer_id': i} for i in range(1,11)]

# read VQA2 json files for train and val
with open(jp(path_vqa2_qa, 'v2_OpenEnded_mscoco_train2014_questions.json'), 'r') as f:
    questions_VQA2_train = json.load(f)

# Get all question ids from VQA2 train
q_ids_train = [e['question_id'] for e in questions_VQA2_train['questions']]

with open(jp(path_vqa2_qa, 'v2_OpenEnded_mscoco_val2014_questions.json'), 'r') as f:
    questions_VQA2_val = json.load(f)

# get all question ids from VQA val
q_ids_val = [e['question_id'] for e in questions_VQA2_val['questions']]

with open(jp(path_vqa2_qa, 'v2_mscoco_train2014_annotations.json'), 'r') as f:
    anns_VQA2_train =json.load(f)

with open(jp(path_vqa2_qa, 'v2_mscoco_val2014_annotations.json'), 'r') as f:
    anns_VQA2_val = json.load(f)

# read COCO-Regions json files for train and val.
with open(jp(path_coco_regions, 'trainqa.json'), 'r') as f:
    qa_COCO_regions_train = json.load(f)

with open(jp(path_coco_regions, 'valqa.json'), 'r') as f:
    qa_COCO_regions_val = json.load(f)

# Add mask name field to all questions from VQA2 (train and val) with 'whole_image_mask.tif'
print('Adding mask field to VQA2 questions')
print('Train...')
for elem in tqdm(questions_VQA2_train['questions']):
    elem['mask_name'] = 'whole_image_mask.tif'
print('Val...')
for elem in tqdm(questions_VQA2_val['questions']):
    elem['mask_name'] = 'whole_image_mask.tif'

# first, add training questions from COCO-Regions
for q_region in tqdm(qa_COCO_regions_train):
    # to solve problem with unicity of ids, check if it's valid, if not create a new one.
    if q_region['question_id'] not in q_ids_train:
        q_id = q_region['question_id']
    else:
        q_id = get_valid_question_id(q_region['question_id'], q_ids_train)

    q_ids_train.append(q_id) # update list of ids

    # add question, image_id, question_id and mask_name to question file
    questions_VQA2_train['questions'].append({  'image_id': get_image_id(q_region['image_name']),
                                                'question': q_region['question'],
                                                'question_id': q_id,
                                                'mask_name': q_region['mask_name']})

    # add remaining info to annotations file
    anns_VQA2_train['annotations'].append({ 'answer_type': 'yes/no',
                                            'multiple_choice_answer': q_region['answer'],
                                            'answers': create_list_of_repeated_answers(q_region['answer']),
                                            'image_id': get_image_id(q_region['image_name']),
                                            'question_type': 'are there',
                                            'question_id': q_id})

# do the same for val
for q_region in tqdm(qa_COCO_regions_val):

    if q_region['question_id'] not in q_ids_val:
        q_id = q_region['question_id']
    else:
        q_id = get_valid_question_id(q_region['question_id'], q_ids_val)

    # update list of ids
    q_ids_val.append(q_id)

    # add question, image_id, question_id and mask_name to question file
    questions_VQA2_val['questions'].append({  'image_id': get_image_id(q_region['image_name']),
                                                'question': q_region['question'],
                                                'question_id': q_region['question_id'],
                                                'mask_name': q_region['mask_name']})

    # add remaining info to annotations file
    anns_VQA2_val['annotations'].append({ 'answer_type': 'yes/no',
                                            'multiple_choice_answer': q_region['answer'],
                                            'answers': create_list_of_repeated_answers(q_region['answer']),
                                            'image_id': get_image_id(q_region['image_name']),
                                            'question_type': 'are there',
                                            'question_id': q_region['question_id']})

# save json files in output folder
print('Saving train questions...')
with open(jp(path_output_qa, 'v2_OpenEnded_mscoco_train2014_questions.json'), 'w') as f:
        json.dump(questions_VQA2_train, f)

print('Saving train anns...')
with open(jp(path_output_qa, 'v2_mscoco_train2014_annotations.json'), 'w') as f:
        json.dump(anns_VQA2_train, f)

print('Saving val questions...')
with open(jp(path_output_qa, 'v2_OpenEnded_mscoco_val2014_questions.json'), 'w') as f:
        json.dump(questions_VQA2_val, f)

print('Saving val anns...')
with open(jp(path_output_qa, 'v2_mscoco_val2014_annotations.json'), 'w') as f:
        json.dump(anns_VQA2_val, f)


print("Don't forget to check unicity of question ids")