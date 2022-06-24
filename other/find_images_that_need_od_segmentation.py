# Project:
#   VQA
# Description:
#   Simple script to find out which images I need to segment before building the DME dataset. 
# Author: 
#   Sergio Tascon-Morales

import os
from os.path import join as jp
import shutil

path_od_available = '/home/sergio814/Documents/PhD/code/data/dme_data_new/masks/OD'

path_images = '/home/sergio814/Documents/PhD/code/data/dme_data_new/images'
images_healthy = os.listdir(jp(path_images, 'healthy'))
images_healthy = [e.split('.')[0] for e in images_healthy] 
images_unhealthy = os.listdir(jp(path_images, 'unhealthy'))
images_unhealthy = [e.split('.')[0] for e in images_unhealthy]

all_images = images_healthy + images_unhealthy

all_od_masks = [e.split('.')[0][:-3] for e in os.listdir(path_od_available)]

done = set(all_images).intersection(set(all_od_masks))

todo = set(all_images) - done

todo_list = list(todo)

# copy files to OD folder so that I can segment them manually
for img in todo_list:
    if img in images_healthy:
        print('Healthy:', img)
        if os.path.exists(jp(path_images, 'healthy', img + '.jpg')):
            shutil.copy(jp(path_images, 'healthy', img + '.jpg'), jp(path_od_available, img + '.jpg'))
        else:
            shutil.copy(jp(path_images, 'healthy', img + '.JPG'), jp(path_od_available, img + '.JPG'))
    elif img in images_unhealthy:
        print('Unhealthy:', img)
        if os.path.exists(jp(path_images, 'unhealthy', img + '.jpg')):
            shutil.copy(jp(path_images, 'unhealthy', img + '.jpg'), jp(path_od_available, img + '.jpg'))
        else:
            shutil.copy(jp(path_images, 'unhealthy', img + '.JPG'), jp(path_od_available, img + '.JPG'))
    else:
        raise ValueError
