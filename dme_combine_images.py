# Project:
#   VQA
# Description:
#   Script to combine images for each subset after creating qa pairs for DME. 
# Author: 
#   Sergio Tascon-Morales

import os
from os.path import join as jp
import pandas as pd
import shutil
from misc import dirs

# paths in origin 
path_anns_dme = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex/annotations/dme.csv'
path_images_dme = '/home/sergio814/Documents/PhD/code/data/dme_dataset_9_balanced/visual/dme_images' # after normalization

# paths in output folder
path_visual = '/home/sergio814/Documents/PhD/code/data/dme_dataset_9_balanced/visual'

def combine_images(path_anns_dme, path_images_dme, path_visual):

    # read csv file with dme grades and subsets
    df_dme = pd.read_csv(path_anns_dme)

    # First, mix healthy and unhealthy images
    for s in ['train', 'val', 'test']:
        print("Now working on", s, 'set...')
        path_subset_combined = jp(path_visual, s)
        dirs.create_folder(path_subset_combined)
        path_healthy = jp(path_visual, 'healthy', s)
        path_unhealthy = jp(path_visual, 'unhealthy', s)
        images_healthy = os.listdir(path_healthy)
        images_unhealthy = os.listdir(path_unhealthy)
        print("Copying healthy images...")
        for img_h in images_healthy:
            shutil.copy(jp(path_healthy, img_h), jp(path_subset_combined, img_h))
        print("Copying unhealthy images...")
        for img_uh in images_unhealthy:
            shutil.copy(jp(path_unhealthy, img_uh), jp(path_subset_combined, img_uh))
        
        print('Copying disease grading images...')
        for i in range(516): # each row (only images from disease grading task of IDRiD, not added images)
            img_name = str(df_dme.loc[i]['image_name']) + '.jpg'
            if str(df_dme.loc[i]['subset']) == s and img_name not in images_healthy and img_name not in images_unhealthy:
                shutil.copy(jp(path_images_dme, img_name), jp(path_subset_combined, img_name))
