# Project:
#   VQA
# Description:
#   Script to convert qa dataset to multi-label dataset so that classifier can be trained
# Author: 
#   Sergio Tascon-Morales

import pickle
from os.path import join as jp
from misc import dirs

#path_processed_data = '/home/sergio814/Documents/PhD/code/data/idrid_single_1_balanced/processed'
path_processed_data = '/home/sergio814/Documents/PhD/code/data/coco_regions_balanced/processed'
path_output = '/home/sergio814/Documents/PhD/code/data/coco_regions_balanced/classif'
dirs.create_folder(path_output)

path_subsets = [jp(path_processed_data, s + 'set.pickle') for s in ['train', 'val']]
path_subsets_out = [jp(path_output, s + 'set.pickle') for s in ['train', 'val']]
path_cats_out = jp(path_output, 'categories.pickle') 

def binarize_answer(answer):
    if answer=='no':
        return 0
    else:
        return 1

def find_entries(mask, data, class_indexes):
    answers = len(class_indexes)*[0]
    for e in data:
        if e['mask_name'] == mask:
            img = e['image_name']
            answers[class_indexes[e['question_tokens'][2]]] = binarize_answer(e['answer'])
    return answers, img



# process train and val sets
for p_in, p_out in zip(path_subsets, path_subsets_out):
    data_new = []
    # read pickle file
    with open(p_in, "rb") as f:
        data = pickle.load(f)
    images_list = list(set(elem['image_name'] for elem in data))
    masks_list = list(set([elem['mask_name'] for elem in data]))
    classes_list = list(set([elem['question_tokens'][2] for elem in data]))

    # generate indexes using train set and use them for val too (guarantee same indexes)
    if 'train' in p_in:
        class_index = {e: i for i, e in enumerate(classes_list)}
        with open(path_cats_out, 'wb') as f:
            pickle.dump(class_index, f)    

    for e in data:
        data_new.append({
                        'image': e['image_name'],
                        'mask': e['mask_name'],
                        'answer': binarize_answer(e['answer']),
                        'index_gt': class_index[e['question_tokens'][2]]
                        })

    with open(p_out, 'wb') as f:
        pickle.dump(data_new, f)



# process test set
with open(jp(path_processed_data, 'testset.pickle'), "rb") as f:
    data = pickle.load(f)
images_list = list(set(elem['image_name'] for elem in data))
masks_list = list(set([elem['mask_name'] for elem in data]))
classes_list = list(set([elem['question_tokens'][2] for elem in data]))
data_new = []
for m in masks_list:
    ans, image = find_entries(m, data, class_index)
    data_new.append({
                    'image': image,
                    'mask': m,
                    'answer': ans
                    })

with open(jp(path_output, 'testset.pickle'), 'wb') as f:
    pickle.dump(data_new, f)