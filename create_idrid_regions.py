# Project:
#   VQA
# Description:
#   Script to create the idrid regions dataset
# Author: 
#   Sergio Tascon-Morales

from dataset_factory.regions_dataset import SingleRegion, ComplementaryRegions
from misc import io

args = io.get_config_file_name()

idrid = SingleRegion(args.path_config, balanced=True)
idrid.build_dataset()