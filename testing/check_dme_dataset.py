# Project:
#   VQA
# Description:
#   Script to visualize random samples from the DME dataset which contains questions about random circular regions as well as questions about whole image and about
#   DME grade.
# Author: 
#   Sergio Tascon-Morales

from os.path import join as jp
import json
import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

dict_suffixes = {'optic disc': 'OD', 'hard exudate': 'EX', 'microaneurysm': 'MA'}

# paths to DME VQA dataset
path_data_vqa = '/home/sergio814/Documents/PhD/code/data/DME_data/dme/dme_dataset_1_balanced'
path_masks_vqa = jp(path_data_vqa, 'masks')
path_qa_vqa = jp(path_data_vqa, 'qa')
path_visual_vqa = jp(path_data_vqa, 'visual')

# paths to ground data
path_data_base = '/home/sergio814/Documents/PhD/code/data/DME_data/new_dme'
path_gt_masks = jp(path_data_base, 'masks')
path_annotations = jp(path_data_base, 'annotations')
path_images = jp(path_data_base, 'images')


def overlay_mask(img, mask, gt, save= False, path_without_ext=None, alpha = 0.7):
    masked = np.ma.masked_where(mask ==0, mask)
    gt = np.ma.masked_where(gt==0, gt)
    fig, ax = plt.subplots()
    ax.imshow(img, 'gray', interpolation='none')
    ax.imshow(masked, 'jet', interpolation='none', alpha=alpha)
    ax.imshow(gt, 'pink', interpolation='none', alpha=alpha)
    fig.set_facecolor("black")
    fig.tight_layout()
    ax.axis('off')
    if save:
        plt.savefig(path_without_ext + '.png', bbox_inches='tight')
    plt.show()


def get_suffix_from_question(q):
    if 'hard' in q:
        suff = 'EX'
    elif 'optic' in q:
        suff = 'OD'
    elif 'microaneurysms' in q:
        suff = 'MA'
    else:
        suff = 'Unknown'

    return suff 


subset = 'train' # where to get random samples from
num_samples_to_show = 10
desired_question_type = 'inside' # question type to get the samples from 

with open(jp(path_qa_vqa, subset + 'qa.json'), 'r') as f:
    data = json.load(f)

curr_q_type = 'dummy'
for i_sample in range(num_samples_to_show):
    while curr_q_type != desired_question_type:
        sample = random.choice(data)
        curr_q_type = sample['question_type']

    image_name_wo_ext = sample['image_name'].split('.')[0]

    # assume image is not healthy
    path_image = jp(path_images, 'not_healthy', subset, sample['image_name'])
    if not os.path.exists(path_image):
        path_image = path_image.replace('not_healthy', 'healthy')

    # read image in original size
    image = Image.open(path_image)
    image_np = np.array(image)

    # read mask and transform it to original size
    path_region = jp(path_masks_vqa, subset, 'maskA', sample['mask_name'])
    mask_small = Image.open(path_region)
    mask_small_np = np.array(mask_small)

    mask = cv2.resize(mask_small_np.astype(np.uint8), dsize=(image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # depending on type of question (question_type), show something different
    if sample['question_type'] == 'inside':
        print('***INSIDE***')
        suffix = get_suffix_from_question(sample['question'])
        path_mask_gt = jp(path_gt_masks, subset, suffix, image_name_wo_ext + '_' + suffix + '.tif')
        if os.path.exists(path_mask_gt):   
            mask_gt = Image.open(path_mask_gt)
            mask_gt_np = np.array(mask_gt)
            if np.count_nonzero(mask_gt*mask) > 0: # * using 1 pixel as threshold
                ans_gt = 'yes'
            else:
                ans_gt = 'no'
        else:
            ans_gt = 'no'
        # show images
        if os.path.exists(path_mask_gt):
            print('Image', image_name_wo_ext, 'question id:', sample['question_id'])
            print('Question:', sample['question'])
            print(' GT ans:', ans_gt)
            print(' Dataset ans:', sample['answer'])
            overlay_mask(image_np, mask, mask_gt_np)
        else: 
            print('Image', image_name_wo_ext, 'should be healthy AF')
        curr_q_type = 'dummy'
        
    elif sample['question_type'] == 'whole':
        print('***WHOLE***')
        suffix = get_suffix_from_question(sample['question'])
        path_mask_gt = jp(path_gt_masks, subset, suffix, image_name_wo_ext + '_' + suffix + '.tif')
        if os.path.exists(path_mask_gt):
            ans_gt = 'yes'
        else:
            ans_gt = 'no'
        print('Image', image_name_wo_ext, 'question id:', sample['question_id'])
        print('Question:', sample['question'])
        print(' GT ans:', ans_gt)
        print(' GT dataset:', sample['answer'])
        plt.imshow(image_np)
        plt.title("GT ans: " + ans_gt + ", ans dataset: " + sample['answer'])
        plt.show()
        curr_q_type = 'dummy'
    else: # grade
        print('***GRADE***')
        path_dme_grades = jp(path_annotations, 'dme_grade.csv')
        df = pd.read_csv(path_dme_grades)
        if sample['answer'] == int(df[df['id'] == image_name_wo_ext]['groundtruth answer']):
            print(sample['question'])
            print(' Correct')
        else:
            print('Problem with the following sample:')
            print(sample)
        curr_q_type = 'dummy'

