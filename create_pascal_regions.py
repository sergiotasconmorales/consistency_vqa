# Project:
#   VQA
# Description:
#   Script to create pascal regions dataset
# Author: 
#   Sergio Tascon-Morales

from dataset_factory.regions_dataset import SingleRegionPascal, ComplementaryRegions
from misc import io

args = io.get_config_file_name()

pascal_regions = SingleRegionPascal(args.path_config, balanced=True)
pascal_regions.divide_train_val(skip=True)
pascal_regions.prepare_masks()
pascal_regions.build_dataset()
