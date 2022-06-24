# Project:
#   VQA
# Description:
#   Script to find inconsistencies between low level perception questions (inside and whole) and reasoning questions (grade) for validation results of DME dataset.
# Author: 
#   Sergio Tascon-Morales

import os 
from os.path import join as jp
import torch 
import json
import pickle
import misc.io as io
from matplotlib import pyplot as plt
from misc import printer
import numpy as np
from PIL import Image
from collections import Counter
from misc import general, dirs
from plot import plot_factory as pf

torch.manual_seed(1234) # use same seed for reproducibility

# read config name from CLI argument --path_config
args = io.get_config_file_name()
plot_inconsistencies = False
check_agreement_fovea_inside = True # whether or not to check how much fovea questions (are there EX in the fovea?) agree with inside questions centered at macula


def get_pred_ans(data, index):
    data_np = data.numpy()
    for i in range(data_np.shape[0]): # each row
        if data_np[i, 0] == index:
            return data_np[i, 1]

def add_predictions_to_qa_dict(data, predictions, mia):
    for e in data:
        question_id = int(e['question_id'])
        ans_pred = get_pred_ans(predictions, question_id)
        e['pred'] = mia[ans_pred]
    return data


def find_and_count_answers(img, q_type, data):
    preds_curr_im_whole = [e['pred'] for e in data if e['image_name']==img and e['question_type'] == q_type]
    counts = Counter(preds_curr_im_whole).most_common() 
    counts_dict = {e[0]: e[1] for e in counts}
    return counts_dict

def find_whole_sample(img, data):
    return [e for e in data if e['image_name'] == img and e['question_type'] == 'whole'][0]


def main():
    # read config file
    config = io.read_config(args.path_config)
    # get config file name
    config_file_name = args.path_config.split('/')[-1].split('.')[0]
    # path to logs
    path_logs = jp(config['logs_dir'], 'idrid_regions_single', config_file_name)
    path_inconsistencies = jp(path_logs, 'inconsistencies')
    dirs.create_folder(path_inconsistencies)
    dirs.create_folders_within_folder(path_inconsistencies, ['zero', 'one', 'two'])

    # path to images and to masks
    path_images = jp(config['path_img'], 'test')
    path_masks = jp(config['path_masks'], 'test', 'maskA')

    path_best_info = jp(path_logs, 'best_checkpoint_info.pt')
    if not os.path.exists(path_best_info):
        raise FileNotFoundError

    # read validation qa pairs
    path_qa = jp(config['path_qa'], 'qa', 'testqa.json')
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
    answers_test_best_epoch = torch.load(jp(path_logs, 'answers', 'answers_epoch_0.pt'))

    qa_pairs_test = add_predictions_to_qa_dict(qa_pairs_test, answers_test_best_epoch, map_index_answer)

    # Now, find potential inconsistencies from Q2 (reasoning correct, perception incorrect)

    print("AGREEMENT BETWEEN FOVEA QUESTIONS AND INSIDE QUESTIONS CENTERED AT FOVEA (for correct inside questions)")
    if check_agreement_fovea_inside:
        # first, list images that have inside questions centered at macula
        images_with_inside_at_macula = [e for e in qa_pairs_test if e['question_type'] == 'inside' and e['center'] == 'macula' and e['answer'] == e['pred']]
        agreement_cnt = 0
        total_cnt = 0
        for img in set([e['image_name'] for e in images_with_inside_at_macula]):
            ans_fovea = [e['pred'] for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'fovea'][0]  # there should be only one
            ans_inside = [e['pred'] for e in images_with_inside_at_macula if e['image_name'] == img] # list because there are two inside questions centered at macula for each image
            # second, for each of the answers about inside, check fovea answer and count number of times agreement happened. 
            for ai in ans_inside:
                total_cnt += 1
                if ans_fovea == ai:
                    agreement_cnt += 1
        print('Agreement rate:', agreement_cnt, '/', total_cnt, '=', '{:.2f}'.format(100*agreement_cnt/total_cnt))

    print("INCONSITENCIES FROM Q2")
    q2_total_sub_right = 0
    q2_total_sub_wrong = 0
    # FIRST, FIND NUMBER OF TRIVIAL INCONSISTENCIES  (GRADE 0)
    printer.print_section('Inconsistencies between 0 grade and low level questions')
    grade_zero_samples = [e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']==e['pred'] and e['answer']==0] # correctly answered questions about grade with answer 0
    print('Model was right about', len(grade_zero_samples), '/', len([e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']==0]), 'grading questions that have answer 0')
    images_zero_grade = list(set([e['image_name'] for e in grade_zero_samples]))
    for q_type in ['inside', 'whole', 'fovea']: # for questions about regions and about whole image
        total_trivial_inconsistencies_zero = 0
        print(q_type.upper(), 'QUESTIONS')
        for img in images_zero_grade:
            # find all samples for current image, which are low level (inside or whole)
            counts_dict = find_and_count_answers(img, q_type, qa_pairs_test)
            if 'yes' in counts_dict:
                print('  ', img, 'has', counts_dict['yes'], 'inconsistencies.')
                total_trivial_inconsistencies_zero += counts_dict['yes']
                q2_total_sub_wrong += counts_dict['yes']
            else:
                print('  ', img, 'has no inconsistencies.')
            if 'no' in counts_dict:
                q2_total_sub_right += counts_dict['no']
        print('There are', total_trivial_inconsistencies_zero, 'trivial inconsistencies for', q_type, 'questions')
        printer.print_line()

    # SECOND, ANALYZE IMAGES FOR WHICH GRADE IS 1 ACCORDING TO GT AND MODEL WAS RIGHT
    printer.print_section('Inconsistencies between 1 grade and low level questions')
    grade_one_samples = [e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']==e['pred'] and e['answer']==1] # correctly answered questions about grade with answer 1
    print('Model was right about', len(grade_one_samples), '/', len([e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']==1]), 'grading questions that have answer 1')
    images_one_grade = list(set([e['image_name'] for e in grade_one_samples]))
    zero_implying_inconsistencies = 0
    for img in images_one_grade:
        # first, check if there are any negatively answered whole questions
        counts_dict = find_and_count_answers(img, 'whole', qa_pairs_test)
        if 'no' in counts_dict:
            print('  ', img, 'has', counts_dict['no'], 'inconsistencies.')
            zero_implying_inconsistencies += counts_dict['no']
            q2_total_sub_wrong += counts_dict['no']
        else:
            print('  ', img, 'has no inconsistencies.')
        if 'yes' in counts_dict:
            q2_total_sub_right += counts_dict['yes']

    print('There are', zero_implying_inconsistencies, 'inconsistencies for whole questions (imply grade 0)')

    # now check for inconsistencies w.r.t. fovea questions
    two_implying_inconsistencies = 0
    for img in images_one_grade:
        counts_dict = find_and_count_answers(img, 'fovea', qa_pairs_test)
        if 'yes' in counts_dict:
            print('  ', img, 'has', counts_dict['yes'], 'inconsistencies.')
            two_implying_inconsistencies += counts_dict['yes']
            q2_total_sub_wrong += counts_dict['yes']
        else:
            print('  ', img, 'has no inconsistencies.')
        if 'no' in counts_dict:
            q2_total_sub_right += counts_dict['no']

    print('There are', two_implying_inconsistencies, 'inconsistencies for fovea questions (imply grade 2)')

    # Now inconsistencies about regions
    two_implying_inconsistencies = 0
    for img in images_one_grade:
        # now check if there are any samples about regions centered at macula that were answered with yes (implies grade 2)
        preds_curr_im_inside_center_macula = [e['pred'] for e in qa_pairs_test if e['image_name']==img and e['center'] == 'macula']
        if len(preds_curr_im_inside_center_macula) < 1:
            continue
        counts2 = Counter(preds_curr_im_inside_center_macula).most_common()
        counts2_dict = {e[0]: e[1] for e in counts2}
        if 'yes' in counts2_dict:
            print('Contradiction found for image', img, '. Region centered at macula with radius < 1 OD diam. was answered with YES (implies grade 2)')
            two_implying_inconsistencies += 1
            q2_total_sub_wrong += counts2_dict['yes']
        if 'no' in counts2_dict:
            q2_total_sub_right += counts2_dict['no']
    print('There are', two_implying_inconsistencies, 'inconsistencies for inside questions centered at macula (imply grade 2)')

    # THIRD, ANALYZE IMAGES FOR WHICH GRADE IS 2 ACCORDING TO GT AND MODEL WAS RIGHT
    printer.print_section('Inconsistencies between 2 grade and low level questions')
    grade_two_samples = [e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']==e['pred'] and e['answer']==2] # correctly answered questions about grade with answer 1
    print('Model was right about', len(grade_two_samples), '/', len([e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']==2]), 'grading questions that have answer 2')
    images_two_grade = list(set([e['image_name'] for e in grade_two_samples]))
    zero_implying_inconsistencies = 0
    for img in images_two_grade:
        counts_dict = find_and_count_answers(img, 'whole', qa_pairs_test)
        if 'no' in counts_dict:
            print('  ', img, 'has', counts_dict['no'], 'inconsistencies.')
            zero_implying_inconsistencies += counts_dict['no']
            q2_total_sub_wrong += counts_dict['no']
        else:
            print('  ', img, 'has no inconsistencies.')
        if 'yes' in counts_dict:
            q2_total_sub_right += counts_dict['yes']

    print('There are', zero_implying_inconsistencies, 'inconsistencies for whole questions (imply grade 0)')

    # now the same for fovea questions (answer should be yes)
    nontwo_implying_inconsistencies = 0
    for img in images_two_grade:
        counts_dict = find_and_count_answers(img, 'fovea', qa_pairs_test) #! why did I count if there is always one fovea question per image?
        if 'no' in counts_dict:
            print('  ', img, 'has', counts_dict['no'], 'inconsistencies.')
            nontwo_implying_inconsistencies += counts_dict['no']
            q2_total_sub_wrong += counts_dict['no']
        else:
            print('  ', img, 'has no inconsistencies.')
        if 'yes' in counts_dict:
            q2_total_sub_right += counts_dict['yes']

    print('There are', nontwo_implying_inconsistencies, 'inconsistencies for fovea questions (imply grade different than 2)')

    nontwo_implying_inconsistencies = 0
    for img in images_two_grade:
        # now check if there are any samples about regions centered at macula that were answered with no (implies grade != 2)
        preds_curr_im_inside_center_macula = [e['pred'] for e in qa_pairs_test if e['image_name']==img and e['center'] == 'macula']
        if len(preds_curr_im_inside_center_macula) < 1:
            continue
        counts2 = Counter(preds_curr_im_inside_center_macula).most_common()
        counts2_dict = {e[0]: e[1] for e in counts2}
        if 'no' in counts2_dict:
            print('   Contradiction found for image', img, '. Region centered at macula with radius < 1 OD diam. was answered with NO (implies grade != 2)')
            nontwo_implying_inconsistencies += 1
            q2_total_sub_wrong += counts2_dict['no'] 
        else:
            print('  ', img, 'has no inconsistencies.')
        if 'yes' in counts2_dict:
            q2_total_sub_right += counts2_dict['yes']
    print('There are', nontwo_implying_inconsistencies, 'inconsistencies for inside questions centered at macula (imply grade different from 2)')

    consistency = 100*(q2_total_sub_right/(q2_total_sub_right + q2_total_sub_wrong))
    print('Total consistency:', "{:.2f}".format(consistency))

    # Now add inconsistency plotting (i38)
    if plot_inconsistencies:
        # Correctly zero-graded images:
        for img in images_zero_grade:
            entry_whole_zero = find_whole_sample(img, qa_pairs_test) # find sample for question about 'whole' image for current image
            entries_inside = [e for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'inside']
            cnt = 0
            for entry in entries_inside:
                if entry['answer'] != entry['pred']: # if question about inside was wrongly answered (inconsistent)
                    image = np.array(Image.open(jp(path_images, img))) # original image
                    mask = np.array(Image.open(jp(path_masks, entry['mask_name'])))
                    pf.plot_inconsistency_dme(image, mask, 0, 0, entry_whole_zero['answer'], entry_whole_zero['pred'], entry['answer'], entry['pred'], save= True, path_without_ext= jp(path_inconsistencies, 'zero', dirs.get_filename_without_extension(img) + '_' + str(cnt)), alpha = 0.3)
                    cnt += 1

        # Correctly one-graded images:
        for img in images_one_grade:
            entry_whole_one = find_whole_sample(img, qa_pairs_test) # find sample for question about 'whole' image for current image
            entries_inside = [e for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'inside']
            cnt = 0
            for entry in entries_inside:
                if entry['answer'] != entry['pred']: # if question about inside was wrongly answered (inconsistent)
                    image = np.array(Image.open(jp(path_images, img))) # original image
                    mask = np.array(Image.open(jp(path_masks, entry['mask_name'])))
                    pf.plot_inconsistency_dme(image, mask, 1, 1, entry_whole_one['answer'], entry_whole_one['pred'], entry['answer'], entry['pred'], save= True, path_without_ext= jp(path_inconsistencies, 'one', dirs.get_filename_without_extension(img) + '_' + str(cnt)), alpha = 0.3)
                    cnt += 1

        # Correctly two-graded images:
        for img in images_two_grade:
            entry_whole_two = find_whole_sample(img, qa_pairs_test) # find sample for question about 'whole' image for current image
            entries_inside = [e for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'inside' and e['center'] == 'macula']
            cnt = 0
            for entry in entries_inside:
                if entry['answer'] != entry['pred']: # if question about inside was wrongly answered (inconsistent)
                    image = np.array(Image.open(jp(path_images, img))) # original image
                    mask = np.array(Image.open(jp(path_masks, entry['mask_name'])))
                    pf.plot_inconsistency_dme(image, mask, 2, 2, entry_whole_two['answer'], entry_whole_two['pred'], entry['answer'], entry['pred'], save= True, path_without_ext= jp(path_inconsistencies, 'two', dirs.get_filename_without_extension(img) + '_' + str(cnt)), alpha = 0.3)
                    cnt += 1

    printer.print_line()

    # Now find inconsistencies from Q3 (reasoning incorrect, perception correct)
    print("INCONSITENCIES FROM Q3")
    q3_global = 0
    # Q3 inconsistencies for grade 0
    q3_wrong_grade_zero_samples = [e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']== 0  and e['pred']!=0] # samples with 0-grade GT that the model graded with something different than 0
    print('Model was wrong about', len(q3_wrong_grade_zero_samples), '/', len([e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']==0]), 'grading questions that have answer 0')
    q3_wrong_images_zero_grade = list(set([e['image_name'] for e in q3_wrong_grade_zero_samples]))
    q3_zero_cnt = 0
    q3_0 = 0
    q3_zero_partial_cnt = 0
    for img in q3_wrong_images_zero_grade: # for ever image
        group = [e['pred'] for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'whole']
        if len(group) == 1:
            ans_whole_zero = group[0] # there should be only one entry
            q3_0 += 1
        else:
            continue # if image does not have a 'whole' answer, go to the next one
        if ans_whole_zero == 'no':
            # check if all inside questions about this image are negative
            # list all answers for inside questions about this image
            ans_inside = [e['pred'] for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'inside']
            if len(list(set(ans_inside))) == 0:
                raise Exception("Ooops")
            if len(list(set(ans_inside))) == 1 and ans_inside[0] == 'no': # if there is only one answer and it's no (meaning all inside questions are negative for this image)
                # now do the same for the question about the fovea
                ans_fovea = [e['pred'] for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'fovea']
                if len(list(set(ans_fovea))) == 1 and ans_fovea[0] == 'no':
                    print(' Image', img, 'has Q3 inconsistencies')
                    q3_zero_cnt += 1
                else:
                    print(' Image', img, 'has partial Q3 inconsistencies (whole and inside questions are right but fovea question is not')
                    q3_zero_partial_cnt += 1
            elif len(list(set(ans_inside))) > 1:
                continue
        else:
            print('Image', img, 'is in Q4')
            continue
    print('Total Q3 reasoning failures for grade 0: ', q3_zero_cnt, 'from', q3_0, 'image considered.')
    print('Total Q3 partial reasoning failures for grade 0: ', q3_zero_partial_cnt)
    printer.print_line()
    
    # Q3 inconsistencies for grade 1
    q3_wrong_grade_one_samples = [e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']== 1  and e['pred']!=1] 
    print('Model was wrong about', len(q3_wrong_grade_one_samples), '/', len([e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']==1]), 'grading questions that have answer 1')
    q3_wrong_images_one_grade = list(set([e['image_name'] for e in q3_wrong_grade_one_samples]))
    q3_one_cnt = 0
    q3_1 = 0
    q3_one_partial_cnt = 0
    for img in q3_wrong_images_one_grade:
        group = [e['answer'] for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'whole'] # there should be only one entry
        if len(group) == 1:
            ans_whole_one = group[0] # there should be only one entry
            q3_1 += 1
        else:
            continue # if image does not have a 'whole' answer, go to the next one
        if ans_whole_one == 'yes':
            # check if inside questions are available about regions centered at macula 
            preds_curr_im_inside_center_macula = [e['pred'] for e in qa_pairs_test if e['image_name'] == img and e['center'] == 'macula']
            if len(preds_curr_im_inside_center_macula) == 0:
                print(' Image', img, 'has partial Q3 inconsistencies (whole is right but no inside centered at macula are available to test further)')
                q3_one_partial_cnt += 1    
            if len(list(set(preds_curr_im_inside_center_macula))) == 1 and preds_curr_im_inside_center_macula[0] == 'no':
                ans_fovea = [e['pred'] for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'fovea']
                if len(list(set(ans_fovea))) == 1 and ans_fovea[0] == 'no':
                    print(' Image', img, 'has Q3 inconsistencies')
                    q3_one_cnt += 1
                else:
                    print(' Image', img, 'has partial Q3 inconsistencies (whole and inside at macula are right but fovea question is wrong)')
                    q3_one_partial_cnt += 1    
        else:
            print('Image', img, 'is in Q4')
            continue
    print('Total Q3 reasoning failures for grade 1: ', q3_one_cnt, 'from', q3_1, 'image considered.')
    print('Total Q3 partial reasoning failures for grade 1: ', q3_one_partial_cnt)
    printer.print_line()

    # Q3 inconsistencies for grade 2
    q3_wrong_images_two_grade = [e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']== 2  and e['pred']!=2]
    print('Model was wrong about', len(q3_wrong_images_two_grade), '/', len([e for e in qa_pairs_test if e['question_type']== 'grade' and e['answer']==2]), 'grading questions that have answer 2')
    q3_wrong_images_two_grade = list(set([e['image_name'] for e in q3_wrong_images_two_grade]))
    q3_two_cnt = 0
    q3_2 = 0 # counts how many images have 'whole' answer. 
    q3_two_partial_cnt = 0
    for img in q3_wrong_images_two_grade:
        group = [e['answer'] for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'whole'] # there should be only one entry
        if len(group) == 1:
            ans_whole_two = group[0] # there should be only one entry
            q3_2 += 1
        else:
            continue # if image does not have a 'whole' answer, go to the next one
        if ans_whole_two == 'yes':
            # check if inside questions are available about regions centered at macula 
            preds_curr_im_inside_center_macula = [e['pred'] for e in qa_pairs_test if e['image_name'] == img and e['center'] == 'macula']
            if len(preds_curr_im_inside_center_macula) == 0:
                print(' Image', img, 'has partial Q3 inconsistencies (whole is right but no inside centered at macula are available to test further)')
                q3_two_partial_cnt += 1                
            if len(list(set(preds_curr_im_inside_center_macula))) == 1 and preds_curr_im_inside_center_macula[0] == 'yes':
                ans_fovea = [e['pred'] for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'fovea']
                if len(list(set(ans_fovea))) == 1 and ans_fovea[0] == 'yes':                
                    print(' Image', img, 'has Q3 inconsistencies')
                    q3_two_cnt += 1
                else:
                    print(' Image', img, 'has partial Q3 inconsistencies (whole and inside at macula are right but fovea question is wrong)')
                    q3_two_partial_cnt += 1 
        else:
            print('Image', img, 'is in Q4')
            continue

    print('Total Q3 reasoning failures for grade 2: ', q3_two_cnt, 'from', q3_2, 'image considered.')
    print('Total Q3 partial reasoning failures for grade 2: ', q3_two_partial_cnt)
    a = 42

if __name__ == '__main__':
    main()