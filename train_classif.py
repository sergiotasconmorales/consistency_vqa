# Project:
#   VQA
# Description:
#   Main train file for classifier
# Author: 
#   Sergio Tascon-Morales


# IMPORTANT: All configurations are made through the yaml config file which is located in config/<dataset>/<file>.yaml. The path to this file is
#           specified using CLI arguments, with --path_config <path_to_yaml_file> . If you don't use comet ml, set the parameter comet_ml to False

import os 
import yaml 
from os.path import join as jp
import comet_ml 
import torch 
import torch.nn as nn 
import misc.io as io
from core.datasets import loaders_factory
from core.models import model_factory
from core.train_vault import criterions, optimizers, train_utils, looper, comet

torch.manual_seed(1234) # use same seed for reproducibility

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)

    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    # load data
    train_loader = loaders_factory.get_classif_loader('train', config, shuffle=True) 
    val_loader = loaders_factory.get_classif_loader('val', config)

    # create model
    model = model_factory.get_classif(config, 87)

    # create criterion
    criterion = criterions.get_criterion(config, device)

    # create optimizer
    optimizer = optimizers.get_optimizer(config, model)

    # initialize experiment
    start_epoch, comet_experiment, early_stopping, logbook, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True, classif=True)

    # create metrics to log
    logbook.log_general_info('config', config)

    # decide which functions are used for training depending on number of possible answers (binary or not)
    train, validate = looper.get_looper_functions(config, classif=True)

    # train loop
    for epoch in range(start_epoch, config['epochs']+1):

        # train for one epoch
        train_epoch_metrics = train(train_loader, model, criterion, optimizer, device, epoch, config, logbook)

        # log training metrics to comet, if required
        comet.log_metrics(comet_experiment, train_epoch_metrics, epoch)

        # validate for one epoch
        val_epoch_metrics, val_results = validate(val_loader, model, criterion, device, epoch, config, logbook)

        # log val metrics to comet, if required
        comet.log_metrics(comet_experiment, val_epoch_metrics, epoch)

        # save validation answers for current epoch
        train_utils.save_results(val_results, epoch, config, path_logs)
        logbook.save_logbook(path_logs)

        early_stopping(val_epoch_metrics, config['metric_to_monitor'], model, optimizer, epoch)

        # if patience was reached, stop train loop
        if early_stopping.early_stop: 
            print("Early stopping")
            break

if __name__ == '__main__':
    main()