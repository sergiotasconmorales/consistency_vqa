# Project:
#   VQA
# Description:
#   Script to check which images have fovea center markups. Basically add column to dme grade csv in which it's indicated if an image has fovea center markup. 
# Author: 
#   Sergio Tascon-Morales

from os.path import join as jp
import pandas as pd
import os
from collections import Counter


PATH_ANNS = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex/annotations'

df_dme = pd.read_csv(jp(PATH_ANNS, 'dme.csv'))
df_fc = pd.read_csv(jp(PATH_ANNS, 'fovea_center_markups.csv'))

images_with_fc = list(df_fc['image_id'])

li = ['no' for k in range(len(df_dme))]

# insert column indicating if image has fovea center markup (assume no for all images at the beginning)
df_dme['has_fovea_center'] = li

for i in range(len(df_dme)):
    image_name = df_dme.iloc[i]['image_name']
    if image_name in images_with_fc:
        df_dme.at[i, 'has_fovea_center'] = 'yes'

# save dataframe
#df_dme.to_csv(jp(PATH_ANNS, 'extended.csv'), index=False)

# get imbalance
for s in ['train', 'val', 'test']:
    print('Subset:', s)
    condi = df_dme[df_dme['has_fovea_center'] == 'yes'][df_dme['subset'] == s]
    condi_grade = list(condi['dme_grade'])
    condi_subset = list(condi['subset'])
    print(Counter(condi_grade).most_common())
    print(Counter(condi_subset).most_common())


# dummy
path_test_unhealthy = '/home/sergio814/Documents/PhD/code/data/dme_data_new/dme_odex/images/unhealthy/test'
images_test_unhealthy = set([e.split('.')[0] for e in os.listdir(path_test_unhealthy)])
possible_to_markup = images_test_unhealthy - set(images_with_fc)
print(possible_to_markup)
