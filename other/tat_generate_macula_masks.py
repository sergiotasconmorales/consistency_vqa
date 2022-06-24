# Project:
#   VQA
# Description:
#   ...
# Author: 
#   Sergio Tascon-Morales

from PIL import Image
import os
from os.path import join as jp
import pandas as pd
import numpy as np
from misc.dirs import create_folder

path_idrid = '/home/sergio814/Documents/PhD/code/data/IDRiD/new_idrid'
path_annotations_macula = '/home/sergio814/Documents/PhD/code/data/sergio_tatiana/DR_IDRiD/Annotations'

path_idrid_images = jp(path_idrid, 'images')
path_idrid_masks = jp(path_idrid, 'masks')

subsets = ['train', 'val', 'test']

# read annotations
df_fc = pd.read_csv(jp(path_annotations_macula, 'IDRiD_Fovea_Center_Markups.csv'))

# for every image of every set, generate mask for fovea center (FC)
for s in subsets:
    print("Processing", s, 'set...')
    images_curr_subset = os.listdir(jp(path_idrid_images, s))
    create_folder(jp(path_idrid_masks, s, 'FC'))
    for img in images_curr_subset:
        img_wo_ext = img.split('.')[0]
        coords = df_fc[df_fc['image_id'] == img_wo_ext]
        if not coords.empty:  # if entry was found, generate mask image
            mask_np = np.zeros_like(np.array(Image.open(jp(path_idrid_images, s, img)))[:,:,0], dtype=np.uint8)
            mask_np[coords['y'], coords['x']] = 1
            mask_pil = Image.fromarray(mask_np)
            # save mask
            mask_pil.save(jp(path_idrid_masks, s, 'FC', img_wo_ext + '_FC.tif'), 'TIFF')