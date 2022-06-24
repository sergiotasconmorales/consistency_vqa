# Project:
#   VQA
# Description:
#   Script to add the qa pairs from VQA-Introspect to VQA2 so that simple model can be trained.
# Author: 
#   Sergio Tascon-Morales
import os
import json
import copy
from tqdm import tqdm
from os.path import join as jp
from misc.printer import print_event as pe


def get_valid_question_id(image_id, all_ids):
    # for current entry, generate next question id under assumption that last 3 digits always contain question index for this image
    for i in range(0, 1000):
        if int(str(image_id) + str(i).zfill(3)) not in all_ids:
            return int(str(image_id) + str(i).zfill(3))


def main():

    path_base = '/home/sergio814/Documents/PhD/code/data/'
    path_vqa2 = jp(path_base, 'VQA2', 'qa')
    path_introspect = jp(path_base, 'VQAIntrospect')
    path_result = jp(path_base, 'VQA2Introspect1')
    if not os.path.exists(path_result):
        os.mkdir(path_result)



    # read VQA2 json files for train and val
    # for each subset do
    for subset in ['train', 'val']: 

        pe('Reading VQA2 data for ' + subset + ' set')
        with open(jp(path_vqa2, 'v2_OpenEnded_mscoco_{}2014_questions.json'.format(subset)), 'r') as f:
            questions_vqa2_json = json.load(f)
        questions_vqa2 = questions_vqa2_json['questions']

        map_question_id_image_id = {e['question_id']:e['image_id'] for e in questions_vqa2}

        # train annotations
        with open(jp(path_vqa2, 'v2_mscoco_{}2014_annotations.json'.format(subset)), 'r') as f:
            anns_vqa2_json = json.load(f)
        anns_vqa2 = anns_vqa2_json['annotations']

        # read QA pairs from VQA-Introspect for train and val
        pe('Reading VQA-Introspect data for ' + subset + ' set')
        with open(jp(path_introspect, 'VQAIntrospect_{}v1.0.json'.format(subset)), 'r') as f:
            introspect_data = json.load(f)

        all_ids = [e['question_id'] for e in questions_vqa2]
        all_ids_temp = copy.deepcopy(all_ids)

        all_ids_main = list(introspect_data.keys())# all question ids for which there are sub-questions (not really)

        #! important: not all entries in introspect have sub-questions, so remove those ids from all_ids_main
        for k,v in introspect_data.items():
            if len(v['introspect']) < 1: # If fake entries
                all_ids_main.remove(k)
            else:   
                cnt = 0
                for p in v['introspect']:
                    cnt += len(p['sub_qa'])
                if cnt == 0: # no sub-questions
                    all_ids_main.remove(k)
                else:
                    pass
            
        
        # before adding new qa pairs, go through VQA data adding roles ot questions
        pe('Adding main and ind tags to VQA2 qa pairs')
        for e in tqdm(anns_vqa2):
            if str(e['question_id']) in all_ids_main:
                e['role'] = 'main'
            else:
                e['role'] = 'ind'

        # add sub-questions to VQA2 files
        for k, v in tqdm(introspect_data.items()):
            # k has the id, v is a dictionary
            if int(k) not in all_ids or k not in all_ids_main:
                continue # for training, because it was made for VQAv1, not all main questions are in the VQA2 data, however, 99.7% are

            for i_sub, sub in enumerate(v['introspect']): # for each available sub-question
                for i_sub_sub, sub_sub in enumerate(sub['sub_qa']):
                    # generate required fields
                    orig_image_id = map_question_id_image_id[int(k)]
                    new_id = get_valid_question_id(orig_image_id, all_ids_temp)
                    assert new_id not in all_ids_temp # sanity check
                    all_ids_temp.append(new_id)
                    question_sub = sub['sub_qa'][i_sub_sub]['sub_question']
                    answer_sub = sub['sub_qa'][i_sub_sub]['sub_answer']
                    answers_sub = [{'answer': sub['sub_qa'][i_sub_sub]['sub_answer'], 'answer_confidence': 'yes', 'answer_id': i} for i in range(1,11)]

                    new_entry_questions = { 'image_id': orig_image_id,
                                            'question': question_sub,
                                            'question_id': new_id,
                                            }

                    new_entry_anns = {  'question_type': 'sub',
                                        'multiple_choice_answer': answer_sub,
                                        'answers': answers_sub,
                                        'image_id': orig_image_id,
                                        'answer_type': 'other',
                                        'question_id': new_id,
                                        'parent': int(k),
                                        'role': 'sub'
                                        }

                    # add entries to vqa qas
                    questions_vqa2.append(new_entry_questions)
                    anns_vqa2.append(new_entry_anns)
        # save in output folder
        with open(jp(path_result, 'v2_OpenEnded_mscoco_{}2014_questions.json'.format(subset)), 'w') as f:
            json.dump(questions_vqa2_json, f)
        
        with open(jp(path_result, 'v2_mscoco_{}2014_annotations.json'.format(subset) ), 'w') as f:
            json.dump(anns_vqa2_json, f)

if __name__ == '__main__':
    main()