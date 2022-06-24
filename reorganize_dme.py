# Project:
#   VQA
# Description:
#   Script to reorganize the dme data (divide into train, val, test) and to bring masks to the same format
# Author: 
#   Sergio Tascon-Morales


import os
from os.path import join as jp
from misc import dirs, printer
import random
import shutil
import pandas as pd
from PIL import Image

random.seed(1234)

# define percentages
percentage_test = 0.2 # of total
percentage_val = 0.2 # of train

# create list for "classes" (lesion types)
list_anomalies = ['EX', 'OD']
mask_images_format = '.tif'

# paths original dataset
path_original = '/home/sergio814/Documents/PhD/code/data/dme_data_new/'
path_images = jp(path_original, 'images')
path_masks = jp(path_original, 'masks')
path_annotations = jp(path_original, 'annotations')
path_disease_grading = jp(path_original, 'dme_images')

file_dme_grade = 'dme.csv'
file_macula_location = 'fovea_center_markups.csv'

# paths for new dataset
path_new = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex_dummy'
path_images_new = jp(path_new, 'images')
path_masks_new = jp(path_new, 'masks')
path_disease_grading_new = jp(path_new, 'dme_images')
path_temp = jp(path_new, 'temp')

# create folder for new dataset
dirs.create_folder(path_new)

# first, bring all images to the same location in a temporal folder
dirs.create_folder(path_temp)
path_images_temp = jp(path_temp, 'images')
dirs.create_folders_within_folder(path_images_temp, ['healthy', 'unhealthy'])
dirs.create_folder(path_images_temp)
path_masks_temp = jp(path_temp, 'masks')
dirs.create_folder(path_masks_temp)

def get_image_name(image_name_with_extension):
    return image_name_with_extension.split(".")[0]

def get_image_format(image_name_with_extension):
    return image_name_with_extension.split(".")[1]

def convert_image_to_tif(path_image, path_target):
    img = Image.open(path_image)
    img.save(jp(path_target, get_image_name(path_image.split("/")[-1])+ '.tif'), 'TIFF')

# copy images
printer.print_section("Copying images...")
for folder in dirs.list_folders(path_images):
    images_names = os.listdir(jp(path_images, folder))
    for img in images_names:
        shutil.copyfile(jp(path_images, folder, img), jp(path_images_temp, folder, img))

# copy masks, converting to tif when necessary
printer.print_section("Copying masks...")
for folder in dirs.list_folders(path_masks):
    for c in os.listdir(jp(path_masks, folder)):
        # check if format is different from tif
        if get_image_format(c) not in ['tif']:
            # open image and save it with tif format
            convert_image_to_tif(jp(path_masks, folder, c), jp(path_masks_temp))
        else:
            mask_path = jp(path_masks, folder, c)
            shutil.copyfile(mask_path, jp(path_masks_temp, c))

# now re-distribute
printer.print_section("Distributing images into train, val, test...")
division = {'healthy': None, 'unhealthy': None}
for c in ['healthy', 'unhealthy']:
    all_images = os.listdir(jp(path_images_temp, c))
    random.shuffle(all_images) # randomly shuffle
    num_images = len(all_images)
    images_train = all_images[:num_images-round(percentage_test*num_images)]
    num_images_train = len(images_train)
    images_test = all_images[num_images-round(percentage_test*num_images):]

    # separate train into train & val
    images_val = images_train[num_images_train-round(percentage_val*num_images_train):]
    images_train = images_train[:num_images_train-round(percentage_val*num_images_train)]

    # move images to folder
    paths_images_trainvaltest = dirs.create_folders_within_folder(jp(path_images_new, c), ['train', 'val', 'test'])
    for img in images_train:
        shutil.copyfile(jp(path_images_temp, c, img), jp(paths_images_trainvaltest[0], img))
    for img in images_val:
        shutil.copyfile(jp(path_images_temp, c, img), jp(paths_images_trainvaltest[1], img))    
    for img in images_test:
        shutil.copyfile(jp(path_images_temp, c, img), jp(paths_images_trainvaltest[2], img))   

    division[c] = {'train': images_train, 'val': images_val, 'test': images_test}

# search for masks and move them too
printer.print_section("Moving masks to corresponding folders...")
paths_masks_trainvaltest = dirs.create_folders_within_folder(path_masks_new, ['train', 'val', 'test'])
for p in paths_masks_trainvaltest:
    _ = dirs.create_folders_within_folder(p, list_anomalies)

all_masks = os.listdir(path_masks_temp)
# copy train masks
for h in ['healthy', 'unhealthy']:
    for subsi in ['train', 'val', 'test']:
        for img in division[h][subsi]:
            image_name = get_image_name(img)
            for c in list_anomalies:
                if os.path.exists(jp(path_masks_temp, image_name + '_' + c + mask_images_format)):
                    shutil.copyfile(jp(path_masks_temp, image_name + '_' + c + mask_images_format), jp(path_masks_new, subsi, c, image_name + '_' + c + mask_images_format))

# remove temp folder
dirs.remove_whole_folder(path_temp)

# finally, copy dme images from the disease grading task
dirs.create_folder(path_disease_grading_new)
images_disease_grading = os.listdir(path_disease_grading)

for img in images_disease_grading:
    shutil.copyfile(jp(path_disease_grading, img), jp(path_disease_grading_new, img))

# copy annotations
path_annotations_new = jp(path_new, 'annotations')
dirs.create_folder(path_annotations_new)
path_dme_grade_anns = jp(path_annotations, file_dme_grade)
df_dme = pd.read_csv(path_dme_grade_anns)
df_dme['subset'] = '' # insert empty column
# iterate dataframe and for each image check if it's in any of the previously dividide datasets
dme_train_samples = 0
dme_val_samples = 0
dme_test_samples = 0
for c in ['healthy', 'unhealthy']:
    for i in range(df_dme.shape[0]):
        if str(df_dme.loc[i]['image_name']) + '.jpg' in division[c]['train']:
            df_dme.at[i, 'subset'] = 'train'
            dme_train_samples += 1
        elif  str(df_dme.loc[i]['image_name']) + '.jpg' in division[c]['val']:
            df_dme.at[i, 'subset'] = 'val'
            dme_val_samples += 1
        elif  str(df_dme.loc[i]['image_name']) + '.jpg' in division[c]['test']:
            df_dme.at[i, 'subset'] = 'test'
            dme_test_samples += 1

a = 42
# now list images in df_dme that do not have a valule in subset column, and divide them 
unassigned_images = []
for i in range(df_dme.shape[0]):
    if df_dme.loc[i]['subset'] == '':
        unassigned_images.append(df_dme.loc[i]['image_name'])


# Now randomly assign images taking into account the amounts that were assigned already.
# First, compute divisions if all images in df_dme were available for assignment
ideal_test = round(df_dme.shape[0]*percentage_test)
ideal_trainval = df_dme.shape[0] - ideal_test
ideal_val = round(ideal_trainval*percentage_val)
ideal_train = ideal_trainval - ideal_val

# from previous numbers, now compute missing images
test_actual_amount = ideal_test - dme_test_samples
train_actual_amount = ideal_train - dme_train_samples
val_actual_amount = ideal_val - dme_val_samples

assert test_actual_amount + train_actual_amount + val_actual_amount == len(unassigned_images)

random.shuffle(unassigned_images)

new_train_images = unassigned_images[:train_actual_amount]
new_val_images = unassigned_images[train_actual_amount: train_actual_amount+val_actual_amount]
new_test_images = unassigned_images[train_actual_amount+val_actual_amount :]

for i in range(df_dme.shape[0]):
    if str(df_dme.loc[i]['image_name']) in new_train_images:
        df_dme.at[i, 'subset'] = 'train'
    elif str(df_dme.loc[i]['image_name']) in new_val_images:
        df_dme.at[i, 'subset'] = 'val'
    elif str(df_dme.loc[i]['image_name']) in new_test_images:
        df_dme.at[i, 'subset'] = 'test'

df_dme.to_csv(jp(path_annotations_new, file_dme_grade), index=False)

# copy annotations about fovea center
path_macula_location_anns = jp(path_annotations, file_macula_location)
shutil.copyfile(path_macula_location_anns, jp(path_annotations_new, file_macula_location))
