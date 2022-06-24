# Project:
#   VQA
# Description:
#   Script to build an image with grades for different locations of an artificial lesion. To test counterfactually how location is taken into account by a "good" model
# Author: 
#   Sergio Tascon-Morales

from PIL import Image 
import numpy as np
from os.path import join as jp

# notation for grades in RGB. Grade 0 is G, grade 1 is B and grade 2 is R
sq_size = 5

# path to grades (stored as .txt)
path_grades = '/home/sergio814/Documents/PhD/code/data/to_inpaint/grade_0/location_test3/grades'

# create dummy image using numpy
grade_map = np.zeros((448,448,3), dtype=np.uint8)

cnt = 0
for i in range(0, 448-32 +1, 16):
    for j in range(0, 448-32 +1, 16):
        #read current image
        with open(jp(path_grades, str(cnt).zfill(4) + '.txt')) as f:
            grade = int(f.readline())

        # i,j is the top left corner so to get the middle I have to add 16
        ii = i+16
        jj = j+16

        if grade == 0:
            # color a small square in the first channel
            grade_map[ii-sq_size:ii+sq_size, jj-sq_size:jj+sq_size, 2] = 255
        elif grade == 1:
            # color with blue a small square in the middle channel
            grade_map[ii-sq_size:ii+sq_size, jj-sq_size:jj+sq_size, 1] = 255
        elif grade == 2:
            # color with red a small square in the last channel
            grade_map[ii-sq_size:ii+sq_size, jj-sq_size:jj+sq_size, 0] = 255
        else:
            raise ValueError("Unknown grade")

        cnt+=1

# save map
img = Image.fromarray(grade_map)
img.save(jp(path_grades, 'grade_map2.png'))

        