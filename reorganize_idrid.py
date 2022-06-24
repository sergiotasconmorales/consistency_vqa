# Project:
#   VQA
# Description:
#   Script to re-organize idrid dataset so that first images are divided into train (80%) and test (20%) and then train is divided into train (80%) and val (20%).
#   Folder structure is kept the same as 
#                                               idrid
#                                                   |-images
#                                                   |     |-train (contains images)
#                                                   |     |-val (contains images)
#                                                   |     |-test (contains images)
#                                                   |-masks
#                                                         |-train
#                                                         |  |-Class1 (contains binary masks for class1)
#                                                         |  |-Class2 (contains binary masks for class2)
#                                                         |  |-Class3 (contains binary masks for class3)
#                                                         |-val
#                                                         |  |-Class1 (contains binary masks for class1)
#                                                         |  |-Class2 (contains binary masks for class2)
#                                                         |  |-Class3 (contains binary masks for class3)
#                                                         |-test
#                                                         |  |-Class1 (contains binary masks for class1)
#                                                         |  |-Class2 (contains binary masks for class2)
#                                                         |  |-Class3 (contains binary masks for class3)
# Author: 
#   Sergio Tascon-Morales

import os
from os.path import join as jp
from misc import dirs
import random
import shutil

random.seed(1234)

# define percentages
percentage_test = 0.2 # of total
percentage_val = 0.2 # of train

# create list for "classes" (lesion types)
list_anomalies = ['EX', 'HE', 'MA', 'OD', 'SE']
mask_images_format = '.tif'

# paths original dataset
path_original = '/home/sergio814/Documents/PhD/code/data/IDRiD/segmentation'
path_images = jp(path_original, 'images')
path_masks = jp(path_original, 'masks')

# paths for new dataset
path_new = '/home/sergio814/Documents/PhD/code/data/IDRiD/new_idrid'
path_images_new = jp(path_new, 'images')
path_masks_new = jp(path_new, 'masks')
path_temp = jp(path_new, 'temp')

# create folder for new dataset
dirs.create_folder(path_new)

# first, bring all images to the same location in a temporal folder
dirs.create_folder(path_temp)
path_images_temp = jp(path_temp, 'images')
dirs.create_folder(path_images_temp)
path_masks_temp = jp(path_temp, 'masks')
dirs.create_folder(path_masks_temp)

def get_image_name(image_name_with_extension):
    return image_name_with_extension.split(".")[0]

# copy images
for folder in os.listdir(path_images):
    images_names = os.listdir(jp(path_images, folder))
    for img in images_names:
        shutil.copyfile(jp(path_images, folder, img), jp(path_images_temp, img))

# copy masks
for folder in os.listdir(path_masks):
    for c in os.listdir(jp(path_masks, folder)):
        masks = os.listdir(jp(path_masks, folder, c))
        for mask in masks:
            shutil.copyfile(jp(path_masks, folder, c, mask), jp(path_masks_temp, mask))

# now re-distribute
all_images = os.listdir(path_images_temp)
random.shuffle(all_images) # randomly shuffle
num_images = len(all_images)
images_train = all_images[:num_images-round(percentage_test*num_images)]
num_images_train = len(images_train)
images_test = all_images[num_images-round(percentage_test*num_images):]
# separate train into train & val
images_val = images_train[num_images_train-round(percentage_val*num_images_train):]
images_train = images_train[:num_images_train-round(percentage_val*num_images_train)]

# move images to folder
paths_images_trainvaltest = dirs.create_folders_within_folder(path_images_new, ['train', 'val', 'test'])
for img in images_train:
    shutil.copyfile(jp(path_images_temp, img), jp(paths_images_trainvaltest[0], img))
for img in images_val:
    shutil.copyfile(jp(path_images_temp, img), jp(paths_images_trainvaltest[1], img))    
for img in images_test:
    shutil.copyfile(jp(path_images_temp, img), jp(paths_images_trainvaltest[2], img))   

# search for masks and move them too
paths_masks_trainvaltest = dirs.create_folders_within_folder(path_masks_new, ['train', 'val', 'test'])
for p in paths_masks_trainvaltest:
    _ = dirs.create_folders_within_folder(p, list_anomalies)

all_masks = os.listdir(path_masks_temp)
# copy train masks
for img in images_train:
    image_name = get_image_name(img)
    for c in list_anomalies:
        if os.path.exists(jp(path_masks_temp, image_name + '_' + c + mask_images_format)):
            shutil.copyfile(jp(path_masks_temp, image_name + '_' + c + mask_images_format), jp(path_masks_new, 'train', c, image_name + '_' + c + mask_images_format))

# copy val masks
for img in images_val:
    image_name = get_image_name(img)
    for c in list_anomalies:
        if os.path.exists(jp(path_masks_temp, image_name + '_' + c + mask_images_format)):
            shutil.copyfile(jp(path_masks_temp, image_name + '_' + c + mask_images_format), jp(path_masks_new, 'val', c, image_name + '_' + c + mask_images_format))

# copy test masks
for img in images_test:
    image_name = get_image_name(img)
    for c in list_anomalies:
        if os.path.exists(jp(path_masks_temp, image_name + '_' + c + mask_images_format)):
            shutil.copyfile(jp(path_masks_temp, image_name + '_' + c + mask_images_format), jp(path_masks_new, 'test', c, image_name + '_' + c + mask_images_format))

# remove temp folder
dirs.remove_whole_folder(path_temp)
