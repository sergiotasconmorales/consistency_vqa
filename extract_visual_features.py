# Project:
#   VQA
# Description:
#   Visual feature (pre) extraction - Extracts features for all images and all sets of a given dataset
# Author: 
#   Sergio Tascon-Morales

import h5py
from misc import io, dirs
from os.path import join as jp
from core.models import model_factory
from core.datasets import loaders_factory

import torch 
import torch.nn as nn 
from torch.autograd import Variable

torch.manual_seed(1234) # use same seed for reproducibility


# read config name from CLI argument --path_config
args = io.get_config_file_name(pre_extract = True)

def main():
    # read config file
    config = io.read_config(args.path_config)
    image_size = config['size']

    model = model_factory.get_visual_model(config)

    dataloader = loaders_factory.get_visual_loader(args.subset, config)

    path_save = jp(config['path_img'], 'extracted')
    path_file = jp(path_save, args.subset + 'set')
    dirs.create_folder(path_save)

    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    extract_features(dataloader, model, path_file, image_size, device)


def extract_features(dataloader, model, path_file, image_size, device):
    # function to extract features from images in dataloader using model and saving them in path_save

    path_features = path_file + '.hdf5'
    path_list = path_file + '.txt'
    features_file = h5py.File(path_features, 'w') # create file to store features

    # make dummi prediction to get size of output
    with torch.no_grad():
        output = model(Variable(torch.ones(1, 3, image_size, image_size)))

    num_images = len(dataloader.dataset)

    # create container for extracted features
    shape_output = (num_images, output.size(1), output.size(2), output.size(3))
    data = features_file.create_dataset('features', shape_output, dtype='f')

    # set evaluation mode
    model.eval()

    index = 0
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            v = Variable(sample['visual'].to(device = device)) # get images for current batch
            output = model(v)
            batch_size = output.size(0)
            data[index:index+batch_size] = output.data.cpu().numpy()
            print("Processing batch ", i+1, "/", len(dataloader))
            index += batch_size
    
    features_file.close()

    # write image names to a txt
    with open(path_list, 'w') as f:
        for img_name in dataloader.dataset.images:
            f.write(img_name + '\n')

if __name__ == '__main__':
    main()