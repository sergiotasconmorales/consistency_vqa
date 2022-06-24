# Project:
#   VQA
# Description:
#   Script to bring the MS coco into the format of IDRiD (pp. 59 of notebook). This means reading all masks for each image
# Author: 
#   Sergio Tascon-Morales

from os.path import join as jp
import json
import numpy as np
from misc import dirs
from PIL import Image
from tqdm import tqdm

from external.coco import coco 

path_images_coco = '/home/sergio814/Documents/PhD/code/data/coco'
path_masks = jp(path_images_coco, 'masks')
dirs.create_folder(path_masks)
path_annotations_coco = '/home/sergio814/Documents/PhD/code/data/coco/annotations_trainval2014/annotations'
prefix_annotations = 'instances'
year = '2014'

subsets = ['train', 'val']

for s in subsets:
    print('Currently processing', s, 'set')
    #annotations_file = jp(path_annotations_coco, prefix_annotations + '_' + s + year + '.json')
    #with open(annotations_file, 'r') as f:
    #    data = json.load(f)
    #a = 42
    c = coco.COCO(annotation_file = jp(path_annotations_coco, prefix_annotations + '_' + s + year + '.json'))
    cat_dict = {elem['id']:elem['name'].replace(" ", "") for elem in c.dataset['categories']}
    img_dict = {elem['id']:elem['file_name'] for elem in c.dataset['images']}
    # create folders in masks folder
    dirs.create_folder(jp(path_masks, s))
    # create folders for each class
    dirs.create_folders_within_folder(jp(path_masks, s), list(cat_dict.values()))
    # iterate through all images, for each (image, category) look for all annotations for that pair, and OR masks of the same category, then save
    for img_id, img_name in tqdm(img_dict.items()):
        for cat_id, cat_name in cat_dict.items():
            ann_ids = c.getAnnIds(imgIds = [img_id], catIds=[cat_id]) # all annotations for current (image, category)
            anns = c.loadAnns(ids = ann_ids)
            image_name_wo_ext = img_name.split(".")[0] # image name without extension
            # get shape of output images
            if len(anns) == 0:
                continue
            m_ref = c.annToMask(anns[0])
            mask_curr = np.zeros_like(m_ref, dtype=np.uint8)
            for a in anns:
                m = c.annToMask(a)
                mask_curr = np.logical_or(mask_curr, m) # do OR on all objects of the same category present in the image
            mask_img = Image.fromarray(255*mask_curr.astype(np.uint8))
            mask_img.save(jp(path_masks, s, cat_name, image_name_wo_ext + "_" + cat_name + '.tif'), 'TIFF')

    """
    # iterate through all annotations to save the masks in the corresponding folders, with the corresponding names
    for ann in tqdm(c.dataset['annotations']):
        curr_cat = cat_dict[ann['category_id']]
        image_name = img_dict[ann['image_id']].split(".")[0] # image name without format
        m = c.annToMask(ann)
        img = Image.fromarray(255*m.astype(np.uint8))
        img.save(jp(path_masks, s, curr_cat, image_name + "_" + curr_cat + '.tif'), 'TIFF')
    """


