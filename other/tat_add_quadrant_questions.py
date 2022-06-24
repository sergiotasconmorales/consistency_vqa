# Project:
#   VQA
# Description:
#   Script to add questions about the quadrants in a balanced way
# Author: 
#   Sergio Tascon-Morales

from dataset_factory.regions_dataset import SingleRegionIdridQuadrants
from misc import io

args = io.get_config_file_name()

idrid = SingleRegionIdridQuadrants(args.path_config, '/home/sergio814/Documents/PhD/code/data/idrid_ODEXMAHEFC_small_bava_balanced', balanced=True)
idrid.add_quadrant_questions()