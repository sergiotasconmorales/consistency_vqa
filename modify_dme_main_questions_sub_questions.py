# Project:
#   VQA
# Description:
#   Script to modify created dataset so that entries contain main questions and also subquestions
# Author: 
#   Sergio Tascon-Morales

from os import symlink
import pickle
from os.path import join as jp
from misc import dirs
from tqdm import tqdm
import shutil
import os

path_base = '/home/sergio814/Documents/PhD/code/data/dme_dataset_6_balanced'
path_data = jp(path_base, 'processed')
path_output = path_base + '_mainsub'
path_output_target = jp(path_output, 'processed')

# create output folder
dirs.create_folder(path_output)
dirs.create_folder(path_output_target)

for s in tqdm(['train', 'val', 'test']): # for each subset
    entries = [] # list for new entries
    # open pickle file
    with open(jp(path_data, s + 'set.pickle'), 'rb') as f:
        data = pickle.load(f)
    # list all images
    all_images = list(set([e['image_name'] for e in data]))
    # for each image, get main (grade) and sub (all others)
    for img in tqdm(all_images):
        q_main = []
        q_sub = []
        # get all questions for current image
        questions_curr_img = [e for e in data if e['image_name'] == img]
        # separate questions into main (grade) and sub (all others)
        for q in questions_curr_img:
            if 'grade' in q['question']:
                q_main.append(q)
            else:
                q_sub.append(q)
        if len(q_main) != 1: # some images do not have main (grade) question
            continue
        # now add len(q_sub) entries to <entries>, where every entry has one main question and one sub-question
        dict_main = {'main_' + k: v for k,v in q_main[0].items() if 'image_name' not in k}
        dict_main['image_name'] = q_main[0]['image_name'] # add image name (same for both questions) without any prefix
        for sub in q_sub:
            dict_sub = {'sub_' + k: v for k,v in sub.items() if 'image_name' not in k}
            # append entry
            entries.append({**dict_main, **dict_sub})
            #? Do I need to set a single id for the entry? Probably not, if so, do it here. 

    print('Total samples for', s, 'set:', len(entries))

    # save new set
    with open(jp(path_output_target, s + 'set.pickle'), 'wb') as f:
        pickle.dump(entries, f)

# finally, copy files from original dataset to new one
folders_to_copy = dirs.list_folders(path_base)
folders_to_copy.remove('processed')
print("Copying files")
for f in tqdm(folders_to_copy):
    shutil.copytree(jp(path_base, f), jp(path_output, f), symlinks=False, ignore=None, copy_function=shutil.copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False)
# copy map files
files_processed = [e for e in os.listdir(path_data) if 'map' in e]
for fi in files_processed:
    shutil.copy(jp(path_data, fi), jp(path_output_target, fi))