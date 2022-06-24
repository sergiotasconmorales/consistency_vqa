# Project:
#   VQA
# Description:
#   Contains function for getting a VQA model trained on the IDRiD dataset, and which is ready for inference
# Author: 
#   Sergio Tascon-Morales

import os 
import yaml 
from os.path import join as jp
import comet_ml
import time 
import torch 
import torch.nn as nn 
import misc.io as io
import pickle
from core.models import model_factory

torch.manual_seed(1234)

def get_vqa(path_config, path_weights_and_vocabs):
    """Creates a VQA model and loads weights"""

    # read config file
    config = io.read_config(path_config)

    # read vocabs
    with open(jp(path_weights_and_vocabs, 'map_index_word.pickle'), 'rb') as f:
        vocab_words = pickle.load(f)

    with open(jp(path_weights_and_vocabs, 'map_index_answer.pickle'), 'rb') as f:
        vocab_answers = pickle.load(f)

    # create model
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    # load weights
    model.module.load_state_dict(torch.load(jp(path_weights_and_vocabs, 'best_checkpoint_model.pt')))

    # set eval mode
    model.eval()

    return model

if __name__ == '__main__':
    m = get_vqa('data/temp/config_039.yaml', 'data/temp')
    print(m)