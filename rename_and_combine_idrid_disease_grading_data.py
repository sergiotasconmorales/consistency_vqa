# Project:
#   VQA
# Description:
#   Script to combine and rename disease grading data of the IDRiD dataset. Basically test images are renamed to continue sequence of training images names.
# Author: 
#   Sergio Tascon-Morales

from misc import dirs
from os.path import join as jp
import os
import shutil
import pandas as pd

path_data = '/home/sergio814/Documents/PhD/code/data/IDRiD/disease_grading'
path_images = jp(path_data, 'imgs')
path_anns = jp(path_data, 'gt')
path_output = jp(path_data, 'disease_grading')

# create output folder
dirs.create_folder(path_output)
dirs.create_folders_within_folder(path_output, ['images', 'annotations'])

path_images_output = jp(path_output, 'images')
path_anns_output = jp(path_output, 'annotations')

# now copy train images to output images folder
images_train = os.listdir(jp(path_images, 'train'))
images_train.sort()
pivot_index = len(images_train) + 1 # index to continue when renaming test images
for img in images_train:
    shutil.copy(jp(path_images, 'train', img), jp(path_images_output, img))

# copy test images but rename them 
images_test = os.listdir(jp(path_images, 'test'))
images_test.sort()
dict_names_old_new = {} # dictionary to save map from old names to new
cnt = pivot_index
for img in images_test:
    shutil.copy(jp(path_images, 'test', img), jp(path_images_output, 'IDRiD_' + str(cnt) + '.jpg'))
    dict_names_old_new[img] = 'IDRiD_' + str(cnt) + '.jpg'
    cnt += 1

# now process annotations
train_anns = pd.read_csv(jp(path_anns, 'train.csv'))
test_anns = pd.read_csv(jp(path_anns, 'test.csv'))

# create new dataframe to put everything
new_pd = pd.DataFrame(columns=['old_name', 'old_subset', 'new_name', 'dme_grade'])

# insert training info to new_pd
for i_row in range(train_anns.shape[0]):
    new_pd.loc[i_row] = [train_anns.loc[i_row]['Image name'],
                        'train',
                        train_anns.loc[i_row]['Image name'], # name did not change for this set
                        train_anns.loc[i_row]['Risk of macular edema ']]

# insert test info 
for i_row in range(test_anns.shape[0]):
    img_name_wo_ext = test_anns.loc[i_row]['Image name']
    new_pd.loc[pivot_index-1] = [img_name_wo_ext,
                                'test',
                                dict_names_old_new[img_name_wo_ext + '.jpg'].split('.')[0],
                                test_anns.loc[i_row]['Risk of macular edema ']]
    pivot_index += 1

new_pd.to_csv(jp(path_anns_output, 'dme.csv'), index=False)