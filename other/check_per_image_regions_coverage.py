# Project:
#   VQA
# Description:
#   Script to check how much of the image is covered by the generated random regions (to see if increasing the number of regions is worth it)
# Author: 
#   Sergio Tascon-Morales

from PIL import Image 
import numpy as np
import json
from os.path import join as jp

path_qa_train = '/home/sergio814/Documents/PhD/code/data/idrid_single_1_balanced/qa'
path_masks = '/home/sergio814/Documents/PhD/code/data/idrid_single_1_balanced/masks/train/maskA'
path_result = '/home/sergio814/Documents/PhD/code/data/idrid_single_1_balanced/overlapped_masks'

# open qa file
with open(jp(path_qa_train, 'trainqa.json'), 'r') as f:
    qa = json.load(f)

images_names = list(set([elem['image_name'] for elem in qa]))

classes_names = list(set([elem['question'].split(" ")[2] for elem in qa]))

dicti_images = {p:[] for p in images_names}
dicti_classes = {p:{c:[] for c in classes_names} for p in images_names}

# for every train image, list all masks that refer to it
for elem in qa:
    dicti_images[elem['image_name']].append(elem['mask_name'])
    dicti_classes[elem['image_name']][elem['question'].split(" ")[2]].append(elem['mask_name'])

# generate OR mask with all masks for every image and for every class
for image_name, classes in dicti_classes.items():
    image_mask = np.zeros((448,448))
    image_name_without_ext = image_name.split('.')[0]
    for class_name, masks in classes.items():
        image_class = np.zeros((448,448))
        for mask in masks:
            # read mask
            img_np = np.array(Image.open(jp(path_masks, mask)))
            image_mask = np.logical_or(image_mask.astype(np.bool), img_np.astype(np.bool))
            image_class = np.logical_or(image_class.astype(np.bool), img_np.astype(np.bool))
        # save image for class
        to_save_class = Image.fromarray(255*image_class.astype(np.uint8))
        to_save_class.save(jp(path_result, image_name_without_ext + '_' + class_name + '.png'), 'PNG')
    # save OR mask for current image
    to_save_image = Image.fromarray(255*image_mask.astype(np.uint8))
    to_save_image.save(jp(path_result, image_name_without_ext + '.png'), 'PNG')


