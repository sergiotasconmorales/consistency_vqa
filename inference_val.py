# Project:
#   VQA
# Description:
#   Inference script for val only
# Author: 
#   Sergio Tascon-Morales

import os 
import yaml 
from os.path import join as jp
import comet_ml 
import numpy as np
import random
import torch 
import torch.nn as nn 
import misc.io as io
from misc import printer
from core.datasets import loaders_factory
from core.models import model_factory
from core.train_vault import criterions, optimizers, train_utils, looper, comet


# read config name from CLI argument --path_config
args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)

    config['train_from'] = 'best' # set this parameter to best so that best model is loaded for validation part
    config['comet_ml'] = False

    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    # load data
    train_loader, vocab_words, vocab_answers, index_unk_answer = loaders_factory.get_vqa_loader('train', config)
    
    val_loader = loaders_factory.get_vqa_loader('val', config)

    # create model
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    if 'weighted_loss' in config:
        if config['weighted_loss']:
            answer_weights = io.read_weights(config) # if use of weights is required, read them from folder where they were previously saved using compute_answer_weights scripts
        else:
            answer_weights = None # If false, just set variable to None
    else:
        answer_weights = None

    # create criterion
    criterion = criterions.get_criterion(config, device, ignore_index = index_unk_answer, weights=answer_weights)

    consisterm = criterions.ConsistencyLossTerm(config)

    # create optimizer
    optimizer = optimizers.get_optimizer(config, model)

    best_epoch, _, _, _, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)

    # decide which functions are used for training depending on number of possible answers (binary or not)
    _, validate = looper.get_looper_functions(config)


    printer.print_section('Inference with best model')


    # Infer val set (again, for debugging purposes)
    metrics_val, results_val = validate(val_loader, model, criterion, device, 2000, config, None, consistency_term=consisterm)
    print("Metrics after inference on the val set, best epoch")
    print(metrics_val)
    train_utils.save_results(results_val, 2000, config, path_logs)

    printer.print_line()

    # produce results for the train data too, for the best epoch
    metrics_train, results_train = validate(train_loader, model, criterion, device, 1000, config, None, consistency_term=consisterm)
    print("Metrics after inference on the train set, best epoch")
    print(metrics_train)
    train_utils.save_results(results_train, 1000, config, path_logs)

    printer.print_section('Inference with last model')

    # produce results for the train data, for the last epoch
    config['train_from'] = 'last'
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    # create criterion
    criterion = criterions.get_criterion(config, device, ignore_index = index_unk_answer)

    # create optimizer
    optimizer = optimizers.get_optimizer(config, model)

    # update model's weights
    best_epoch, _, _, _, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)

    metrics_val, results_val = validate(val_loader, model, criterion, device, 2001, config, None, consistency_term=consisterm)
    print("Metrics after inference on the val set, last epoch")
    print(metrics_val)
    train_utils.save_results(results_val, 2001, config, path_logs)

    printer.print_line()

    metrics_train, results_train = validate(train_loader, model, criterion, device, 1001, config, None, consistency_term=consisterm)
    print("Metrics after inference on the train set, last epoch")
    print(metrics_train)
    train_utils.save_results(results_train, 1001, config, path_logs)

if __name__ == '__main__':
    main()
