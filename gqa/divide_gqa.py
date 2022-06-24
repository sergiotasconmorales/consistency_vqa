# Project:
#   VQA
# Description:
#   Script for dividing GQA images into train, val and test
# Author: 
#   Sergio Tascon-Morales

import os
import random
import shutil
import pandas as pd
from tqdm import tqdm
from os.path import join as jp


qa_pairs_filename = 'temp_binary.csv'
qa_pairs_output_filename = 'qa.csv'
path_qa_base = '/home/sergio814/Documents/PhD/code/data/GQA/questions1.2/'
path_qa_pairs = path_qa_base + qa_pairs_filename
path_images = '/home/sergio814/Documents/PhD/code/data/GQA/images'
subsets = ['train', 'val', 'test']

def list_files(path):
    return [k for k in os.listdir(path) if not os.path.isdir(jp(path, k))]

def create_folder(path):
    # creation of folders
    if not os.path.exists(path):
        try:
            os.mkdir(path) # try to create folder
        except:
            os.makedirs(path) # create full path

def add_subset_to_df(df, images):
    # iterate over df
    sub = []
    prev_img = 'dummy'
    for i_row in tqdm(range(df.shape[0])):
        curr_img = str(df.iloc[i_row]['img_id']) + '.jpg'
        if curr_img == prev_img:
            sub.append(s)
            continue
        for s in list(images.keys()):
            if curr_img in images[s]:
                sub.append(s)
                prev_img = curr_img
                break
    df['subset'] = sub
    return df

# read df
df = pd.read_csv(path_qa_pairs)

# list all images
all_images = list_files(path_images)

if bool(all_images): # if list not empty
    random.shuffle(all_images) # just in case
    images = {}
    # randomly divide images into train, val and test
    pivot1 = int(0.8*len(all_images))
    trainval_images = all_images[:pivot1]
    images['test'] = all_images[pivot1:]
    assert len(trainval_images) + len(images['test']) == len(all_images)
    pivot2 = int(0.8*len(trainval_images))
    images['train'] = trainval_images[:pivot2]
    images['val'] = trainval_images[pivot2:]
    assert len(images['train']) + len(images['val']) == len(trainval_images)

    # put images in corresponding folders
    _ = [create_folder(jp(path_images, s)) for s in subsets]
    for s in subsets:
        for img in images[s]:
            shutil.move(jp(path_images, img), jp(path_images, s, img))

    # invoke function to add subset info to df
    df = add_subset_to_df(df, images)
    # save
    df.to_csv(jp(path_qa_base, qa_pairs_output_filename), index=False)

else: 
    images = {}
    # list files again
    for s in subsets:
        images[s] = list_files(jp(path_images, s))

    # invoke function to add subset info to df
    df = add_subset_to_df(df, images)
    # save
    df.to_csv(jp(path_qa_base, qa_pairs_output_filename), index=False)