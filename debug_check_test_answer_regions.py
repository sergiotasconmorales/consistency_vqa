# Project:
#   VQA
# Description:
#   Script for checking, on the test set, for every image, for every type of question, which type of windows cause YES and NO answers from the model.
# Author: 
#   Sergio Tascon-Morales

from os.path import join as jp
import os
import torch
import pickle
from misc import dirs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

config_file_name = 'config_014'
invert = False
path_logs = '/home/sergio814/Documents/PhD/code/logs/idrid_regions_single/'
path_gt = '/home/sergio814/Documents/PhD/code/data/IDRiD/new_idrid/masks/test'
dict_suffixes = {'soft': 'SE', 'hard': 'EX', 'microaneurysms': 'MA', 'hemorrhages': 'HE', 'optic':'OD'}

path_dataset = '/home/sergio814/Documents/PhD/code/data/idrid_OD_balanced'
path_visual = jp(path_dataset, 'visual', 'test')
path_masks = jp(path_dataset, 'masks', 'test', 'maskA')

# create path to logs
path_logs_config = jp(path_logs, config_file_name)
path_results = jp(path_logs_config, 'test_mask_boxes')
dirs.create_folder(path_results)
path_answers = jp(path_logs_config, 'answers')
path_answers_test = jp(path_answers, 'answers_epoch_0.pt')

# path to processed qa pairs
path_qa = jp(path_dataset, 'processed', 'testset.pickle')


def get_image_and_mask_and_biomarker_names(data, question_id):
    for e in data:
        if e['question_id'] == question_id:
            mask_name = e['mask_name']
            image_name = e['image_name']
            biomarker = e['question_tokens'][2]
            break

    return image_name, mask_name, biomarker


def overlay_mask(img, mask, gt, path_without_ext, alpha = 0.7):
    masked = np.ma.masked_where(mask ==0, mask)
    gt = np.ma.masked_where(gt==0, gt)
    fig, ax = plt.subplots()
    ax.imshow(img, 'gray', interpolation='none')
    ax.imshow(masked, 'jet', interpolation='none', alpha=alpha)
    ax.imshow(gt, 'pink', interpolation='none', alpha=alpha)
    fig.set_facecolor("black")
    fig.tight_layout()
    ax.axis('off')
    plt.savefig(path_without_ext + '.png', bbox_inches='tight')
    plt.close()

def remove_extension(name):

    return name.split(".")[0]

# check that inference was made for test set
if not os.path.exists(path_answers_test):
    raise FileNotFoundError

# read answers file
answers = torch.load(path_answers_test)

# read qa pairs
with open(path_qa, 'rb') as f:
    data = pickle.load(f)

# list of images and masks (for creating folders)
all_images = list(set([e['image_name'].split(".")[0] for e in data]))
dirs.create_folders_within_folder(path_results, all_images)
all_classes = list(set([e['question_tokens'][2] for e in data]))

for img in all_images:
    dirs.create_folders_within_folder(jp(path_results, img), all_classes)

for img in all_images:
    for c in all_classes:
        dirs.create_folders_within_folder(jp(path_results, img, c), ['yes', 'no'])


# create folders for results

rex = remove_extension
# iterate through all test questions
for i in tqdm(range(answers['results'].shape[0])):
    q_id = answers['results'][i, 0].item()
    ans_model = answers['results'][i,1].item()
    ans_gt = int(answers['answers'][i,0].item())
    image_name, mask_name, biomarker = get_image_and_mask_and_biomarker_names(data, q_id)
    
    # read image and mask
    image = np.array(Image.open(jp(path_visual, image_name)))
    mask = np.array(Image.open(jp(path_masks, mask_name)))

    # read gt mask
    path_gt_mask = jp(path_gt, dict_suffixes[biomarker], image_name.split(".")[0] + '_' + dict_suffixes[biomarker] + '.tif')
    if os.path.exists(path_gt_mask):
        gt_mask = Image.open(path_gt_mask).resize((448,448), resample = Image.NEAREST)
        gt_mask = np.array(gt_mask)
    else:
        gt_mask = np.zeros_like(image)

    # define path depending on answer
    if not invert:
        if ans_model == 1:
            path_target = jp(path_results, rex(image_name), biomarker, 'no', rex(mask_name))
        else:
            path_target = jp(path_results, rex(image_name), biomarker, 'yes', rex(mask_name))
    else:
        if ans_model == 1:
            path_target = jp(path_results, rex(image_name), biomarker, 'yes', rex(mask_name))
        else:
            path_target = jp(path_results, rex(image_name), biomarker, 'no', rex(mask_name))

    # add a mark depending on GT answer
    if not invert:
        if ans_gt>0:
            image[:50,-50:,2] = 255 # green is positive
        else:
            image[:50,-50:,1] = 255 # blue is negative
    else:
        if ans_gt>0:
            image[:50,-50:,1] = 255 # green is positive
        else:
            image[:50,-50:,2] = 255 # blue is negative

    overlay_mask(image, mask, gt_mask, path_target, alpha=0.4)

    # save image with region 



