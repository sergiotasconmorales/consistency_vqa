# Project:
#   VQA
# Description:
#   Script to find which images still need to be segmented for the test set of DME
# Author: 
#   Sergio Tascon-Morales

import os
import shutil
import pandas as pd
from os.path import join as jp

path_od_masks_test = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex/masks/test/OD'
path_dme_anns = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex/annotations'
path_dme_images = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex/dme_images'

df = pd.read_csv(jp(path_dme_anns, 'dme.csv'))

test_images_wo_ext = set(df[df['subset'] == 'test']['image_name'].values.tolist())
#test_images_wo_ext = set(df[(df['subset'] == 'test') & (df['dme_grade'] == 0)]['image_name'].values.tolist())
available_od_masks = set([e[:-7] for e in os.listdir(path_od_masks_test)])

missing = test_images_wo_ext - available_od_masks
 
print(len(missing))

# now copy missing images to path-od_masks_test
#for img in missing:
#    shutil.copy(jp(path_dme_images, img + '.jpg'), jp(path_od_masks_test, 'todo', img + '.jpg'))