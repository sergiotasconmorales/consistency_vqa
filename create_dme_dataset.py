# Project:
#   VQA
# Description:
#   Creation of the DME dataset, which contains questions about DME grade as well as about presence of hard exudates in circular regions and in the whole image
# Author: 
#   Sergio Tascon-Morales

from dataset_factory.regions_dataset import DMEDataset2
from misc import io
from os.path import join as jp
from dme_combine_images import combine_images

args = io.get_config_file_name()

idrid = DMEDataset2(args.path_config, balanced=True)
idrid.build_dataset()

# After creating dataset, combine images
path_dme_images  = jp(idrid.path_images_output, 'dme_images')
path_dme_anns = jp(idrid.path_anns, idrid.config['dme_filename'])
path_visual = idrid.path_images_output

combine_images(path_dme_anns, path_dme_images, path_visual)

