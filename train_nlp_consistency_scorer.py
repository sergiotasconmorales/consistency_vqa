# Project:
#   VQA
# Description:
#   Script to create and train a consistency scorer, which is an NLP model that receives two qa pairs and scores their consistency. 
# Author: 
#   Sergio Tascon-Morales

import comet_ml
import torch
from os.path import join as jp
from core.datasets import loaders_factory
from core.models import model_factory
from torch.optim.lr_scheduler import ReduceLROnPlateau
import misc.io as io
from core.train_vault import criterions, optimizers, train_utils, looper, comet

args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)

    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    # now read data and encode questions
    train_loader = loaders_factory.get_nlp_loader('train', config, shuffle=True)
    val_loader = loaders_factory.get_nlp_loader('val', config, shuffle=False)

    model = model_factory.get_nlp_model(config, train_loader.dataset.vocab_words)

    criterion = criterions.get_criterion(config, device)

    optimizer = optimizers.get_optimizer(config, model)

    # create LR scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    start_epoch, comet_experiment, early_stopping, logbook, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)

    logbook.log_general_info('config', config)

    train, validate = looper.get_looper_functions(config) # TODO

    for epoch in range(start_epoch, config['epochs']+1):
        
        train_metrics = train(model, train_loader, criterion, optimizer, epoch, device)
        comet.log_metrics(comet_experiment, train_metrics, epoch)

        val_metrics = validate(model, val_loader, epoch, criterion, device)
        comet.log_metrics(comet_experiment, val_metrics, epoch)

        scheduler.step(val_metrics[config['metric_to_monitor']])

        #train_utils.save_results(val_results, epoch, config, path_logs)
        logbook.save_logbook(path_logs)

        early_stopping(val_metrics, config['metric_to_monitor'], model, optimizer, epoch)

        # if patience was reached, stop train loop
        if early_stopping.early_stop: 
            print("Early stopping")
            break      

if __name__ == '__main__':
    main()