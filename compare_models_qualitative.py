# Project:
#   VQA
# Description:
#   Script to compare models. Basically an image (or a set of images) is chosen (randomly or not) and then all the questions for that image are plotted
#   in terms of the windows on the image. Colors are used to indicate if the model was right or not. 
# Author: 
#   Sergio Tascon-Morales

from os.path import join as jp
import pickle
import random
from PIL import Image
import numpy as np
import torch
from plot import plot_factory as pf
from tqdm import tqdm

NUM_IMAGES = 50
images_to_check = []

path_logs = '/home/sergio814/Documents/PhD/code/logs/coco_regions'
path_data = '/home/sergio814/Documents/PhD/code/data/coco_regions_balanced'
path_images = jp(path_data, 'visual',  'val')
path_masks = jp(path_data, 'masks', 'val', 'maskA')

if images_to_check:
    num_images = len(images_to_check)
else:
    num_images = NUM_IMAGES

models_to_compare = ['cocosingle_config_009', 'cocosingle_config_008', 'cocosingle_config_005', 'cocosingle_config_007']

# read validation set
path_val_data = jp(path_data, 'processed', 'valset.pickle')
with open(path_val_data, 'rb') as f:
    data = pickle.load(f)

all_val_images = list(set([e['image_name'] for e in data]))

for i in range(num_images): # for every of the desired images
    print("Processing image number", i+1, "/", num_images)
    if not images_to_check: # if no images given, choose randomly
        curr_img = random.choice(all_val_images)
        all_val_images.remove(curr_img)

        # list all entries for current image, then all categories, then randomly choose a category
        entries_img = [e for e in data if e['image_name'] == curr_img]
        # list categories present in the image
        cats = list(set([e['question_tokens_with_UNK'][2] for e in entries_img]))
        cat = random.choice(cats) # randomly choose a category
        entries_img_cat = [e for e in entries_img if cat in e['question']]
        entries_img_cat_dict = {e['question_id']: e['answer_index'] for e in entries_img_cat}
        entries_img_cat_dict_masks = {e['question_id']: e['mask_name'] for e in entries_img_cat}
        ids_to_plot = [e['question_id'] for e in entries_img_cat]

        # read current image
        img = Image.open(jp(path_images, curr_img))
        img_array = np.array(img)

        # for each of the models (config files), read the answers provided by the model
        for i_mod, config in enumerate(models_to_compare):
            print(">> Model ", i_mod+1, '/', len(models_to_compare))
            path_info_best_epoch = jp(path_logs, config, 'best_checkpoint_info.pt')
            best_epoch = torch.load(path_info_best_epoch)['epoch']
            answers = torch.load(jp(path_logs, config, 'answers', 'answers_epoch_' + str(best_epoch) + '.pt'))
            # now find all images in data which correspond to the current image
            masks = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
            for curr_id in tqdm(ids_to_plot):
                gt_ans = entries_img_cat_dict[curr_id]
                for i_row in range(answers['results'].shape[0]):
                    if answers['results'][i_row][0] == curr_id:
                        m_ans = answers['results'][i_row][1].item()
                        break
                # decide if tp, tn, fp, fn
                mask_array = np.array(Image.open(jp(path_masks, entries_img_cat_dict_masks[curr_id])))
                if gt_ans == 1 and m_ans == 1: # TP
                    masks['TP'].append(mask_array)
                elif gt_ans == 1 and m_ans == 0: # FN
                    masks['FN'].append(mask_array)
                elif gt_ans == 0 and m_ans == 0: #TN
                    masks['TN'].append(mask_array)
                elif gt_ans == 0 and m_ans == 1: # FP
                    masks['FP'].append(mask_array)

            pf.overlay_windows_with_colors(img_array, masks, cat, save=True, path_without_ext=jp(path_logs, config, 'w_' + curr_img[:-4]))