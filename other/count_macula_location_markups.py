# Project:
#   VQA
# Description:
#   Script to get total number of available fovea center markups and corresponding distribution in train, val, test. 
# Author: 
#   Sergio Tascon-Morales

import pandas as pd
from os.path import join as jp

from pandas.io.parsers import count_empty_vals

path_annotations = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex/annotations'
path_dme_grades = jp(path_annotations, 'dme.csv')
path_fovea_centers = jp(path_annotations, 'fovea_center_markups.csv')

df_dme = pd.read_csv(path_dme_grades)
df_fc = pd.read_csv(path_fovea_centers)

def get_subset(df, img_id):
    s = df[df['image_name'] == img_id]['subset']
    return s.item()

print('Total images with fovea center markup:', df_fc.shape[0])

counts = {'train': 0, 'val': 0, 'test': 0}

for i_row in range(df_fc.shape[0]):
    img_id = df_fc.loc[i_row]['image_id']
    subset_curr_img = get_subset(df_dme, img_id)
    counts[subset_curr_img] += 1

print(counts)
counts_dme = df_dme.value_counts(subset=['subset'])
print(counts_dme)

test_0 = df_dme[(df_dme['subset'] == 'test') & (df_dme['dme_grade'] == 0)]['image_name'].values.tolist()
available_fc = df_fc['image_id'].values.tolist()
cnt = 0
todo = []
for e in test_0:    
    if e in available_fc:
        cnt += 1
        print('Image', e, 'OK')
    else:
        print('Image', e, 'does not have markup')
        todo.append(e)
print(cnt, 'images have fovea center markup (out of', len(test_0), '). So try to markup', len(test_0) - cnt, 'images.')
print(todo)