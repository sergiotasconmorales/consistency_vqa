# Project:
#   VQA
# Description:
#   Script for providing models
# Author: 
#   Sergio Tascon-Morales

import torch.nn as nn
from . import models
from .components import image

def get_vqa_model(config, vocab_words, vocab_answers):
    # function to provide a vqa model

    model = getattr(models, config['model'])(config, vocab_words, vocab_answers)

    if config['data_parallel'] and config['cuda']:
        model = nn.DataParallel(model).cuda()

    return model

def get_visual_model(config):
    # create model for visual feature extraction and move it to the gpu if required, after making it parallel, if required
    model = image.get_visual_feature_extractor(config)

    if config['data_parallel'] and config['cuda']:
        model = nn.DataParallel(model).cuda()

    return model

def get_classif(config, num_outputs):

    model = models.Classifier(config, num_outputs)

    if config['data_parallel'] and config['cuda']:
        model = nn.DataParallel(model).cuda()

    return model

def get_nlp_model(config, vocab_words):

    model = getattr(models, 'NLPModel')(   config['word_embedding_size'], 
                                                config['num_layers_LSTM'], 
                                                config['question_feature_size'], 
                                                vocab_words,
                                                config['classifier_hidden_size'],
                                                config['classifier_dropout'])

    if config['data_parallel'] and config['cuda']:
        model = nn.DataParallel(model).cuda()

    return model