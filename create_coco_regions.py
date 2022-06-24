# Project:
#   VQA
# Description:
#   Script for the creation of the region-based version of MSCOCO
# Author: 
#   Sergio Tascon-Morales


from dataset_factory.regions_dataset import SingleRegion, ComplementaryRegions
from misc import io

args = io.get_config_file_name()

coco_regions = SingleRegion(args.path_config, balanced=True)
coco_regions.build_dataset()