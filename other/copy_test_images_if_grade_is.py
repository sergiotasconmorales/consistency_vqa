# Project:
#   VQA
# Description:
#   Script to copy test images with a particular grade, so that they can be manually inpainted.
# Author: 
#   Sergio Tascon-Morales

import shutil
import pandas as pd
from os.path import join as jp
import os

grade = 0 # which grade images should be copied from the test set

path_grades = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex/annotations/dme.csv'
path_images = '/home/sergio814/Documents/PhD/code/data/dme_dataset_8_balanced/visual/test'
path_target = '/home/sergio814/Documents/PhD/code/data/to_inpaint' # where to copy the images

if not os.path.exists(jp(path_target, 'grade_' + str(grade))):
    os.mkdir(jp(path_target, 'grade_' + str(grade)))

# read csv with pandas
df = pd.read_csv(path_grades)

# filter df to desired grade and test set
test_images = df[df['subset'] == 'test']
images_to_copy = list(test_images[test_images['dme_grade'] == grade]['image_name'])

# copy images
for img in images_to_copy:
    img = img + '.jpg'
    shutil.copy(jp(path_images, img), jp(path_target, 'grade_' + str(grade), img))