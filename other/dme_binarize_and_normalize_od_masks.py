# Project:
#   VQA
# Description:
#   Script to binarize and normalize (to 0-1) OD masks so that everything agrees
# Author: 
#   Sergio Tascon-Morales

from operator import sub
import os
import numpy as np
from os.path import join as jp
from PIL import Image
from tqdm import tqdm
import shutil 

path_masks = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex_counterfactual_test/masks'

subsets_dirs = os.listdir(path_masks) #folters test, train, val

for subset in subsets_dirs:
    print("Processing", subset, "set...")
    biomarkers_dirs = os.listdir(jp(path_masks, subset)) # EX, MA, OD
    for bm in biomarkers_dirs:
        print("  Biomarker: ", bm)
        temp_dir = jp(path_masks, subset, bm, 'temp')
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        # list mask images, skipping folders
        masks = [k for k in os.listdir(jp(path_masks, subset, bm)) if not os.path.isdir(jp(path_masks, subset, bm, k))] # list of masks to check for binarity and to bring to (0,1) values
        for m in tqdm(masks):
            img = Image.open(jp(path_masks, subset,  bm, m)) # open image
            img_np = np.array(img)
            # if image has more than two dimensions (some are in RGBA), take only Red channel and discard A
            if len(img_np.shape) > 2:
                img_np = img_np[:,:,0]
            # first, check if image is binary
            if len(np.unique(img_np)) > 2: # if not binary, binarize it
                img_bin = img_np > 0 # binarize by using 0 as threshold. This will be a boolean
                # save as np.uint8
                img_to_save = Image.fromarray(255*img_bin.astype(np.uint8))
                img_to_save.save(jp(temp_dir, m), 'TIFF')
            else: # if already binary, make sure max value is 255
                if np.max(img_np) != 255:
                    img_np = 255*(img_np.astype(float)/np.max(img_np))
                    #save as np.uint8
                    img_to_save = Image.fromarray(img_np.astype(np.uint8))
                    img_to_save.save(jp(temp_dir, m), 'TIFF')
                else: # if already binary and max value is 255, open the image and save it without changing it
                    img_to_save = Image.fromarray(img_np.astype(np.uint8))
                    img_to_save.save(jp(temp_dir, m), 'TIFF')
        # remove original masks and replace them with pre-processed ones
        for m in tqdm(masks):
            os.remove(jp(path_masks, subset, bm, m))
            shutil.move(jp(temp_dir, m), jp(path_masks, subset, bm, m))
        os.rmdir(temp_dir) # remove temp dir
