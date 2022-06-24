# Project:
#   VQA
# Description:
#   Script to apply automatic inpainting
# Author: 
#   Sergio Tascon-Morales

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from os.path import join as jp
from skimage.morphology import disk, binary_dilation
from skimage.restoration import inpaint

radius = 3

path_images = '/home/sergio814/Documents/PhD/code/data/to_inpaint/new/test'
path_results = jp(path_images, 'results_opencv')
if not os.path.exists(path_results):
    os.mkdir(path_results)

# list jpg images to get names
jpg = [e.split('.')[0] for e in os.listdir(path_images) if 'jpg' in e]
for image in jpg:
    image_np = np.array(Image.open(jp(path_images, image + '.jpg')))
    mask_np = np.array(Image.open(jp(path_images, image + '_EX.tif')))
    assert mask_np.ndim == 2 # sanity check
    # dilate mask to improve inpainting results
    dilated_mask = binary_dilation(mask_np.astype(np.uint8), disk(radius, dtype=bool)).astype(np.uint8)
    inpainted = cv.inpaint(image_np, (255*dilated_mask), 3, cv.INPAINT_TELEA)
    #inpainted = inpaint.inpaint_biharmonic(image_np, dilated_mask, channel_axis=-1)
    plt.imsave(jp(path_results, image + '00' + '.jpg'), inpainted)