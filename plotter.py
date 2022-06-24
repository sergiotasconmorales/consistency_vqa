# Project:
#   VQA
# Description:
#   Script to plot things
# Author: 
#   Sergio Tascon-Morales

from os.path import join as jp
import misc.io as io
import torch
import json
import pickle
import os
from plot import plot_factory as pf
from metrics import metrics 
from misc import general 
import matplotlib.pyplot as plt

torch.manual_seed(1234) # use same seed for reproducibility

# read config name from CLI argument --path_config
args = io.get_config_file_name()


def main():
    # read config file
    config = io.read_config(args.path_config)

    config_file_name = args.path_config.split("/")[-1].split(".")[0]

    path_logs = jp(config['logs_dir'], config['dataset'], config_file_name)

    # first, plot logged learning curves for all available metrics
    with open(jp(path_logs, 'logbook.json'), 'r') as f:
        logbook = json.load(f)

    general_info = logbook['general']
    train_metrics = logbook['train']
    val_metrics = logbook['val']

    #* assumption: all reported train metrics were also reported for validation

    for (k_train, v_train), (k_val, v_val) in zip(train_metrics.items(), val_metrics.items()):
        assert k_train.split('_')[0] == k_val.split('_')[0] # check that metrics correspond
        metric_name = k_train.split('_')[0]

        pf.plot_learning_curve(v_train, v_val, metric_name, title=general_info['config']['model'] + ' ' + config_file_name, save=True, path=path_logs)



    # if model is binary, plot ROC and PRC curves along with AUC and AP
    if config['num_answers'] == 2:
        # first, generate plots for best validation epoch
        best_epoch_info_path = jp(path_logs, 'best_checkpoint_info.pt')
        best_epoch_info = torch.load(best_epoch_info_path, map_location=torch.device('cpu'))

        best_epoch_index = best_epoch_info['epoch']
        # now go to answers folder and read info from there
        path_val_answers_file = jp(path_logs, 'answers', 'answers_epoch_' + str(best_epoch_index) + '.pt')
        answers_best_val_epoch = torch.load(path_val_answers_file, map_location=torch.device('cpu')) # dictionary with keys: results, answers. results contains tensor with (question_index, model's answer), answers is  (target, prob)

        auc_val, ap_val, roc_val, prc_val = metrics.compute_roc_prc(answers_best_val_epoch['answers'])
        pf.plot_roc_prc(roc_val, auc_val, prc_val, ap_val, title='Validation best epoch ('+ str(best_epoch_index) + ')', save=True, path=path_logs, suffix='val')

        # for generation of biomarker-wise metrics, read data
        path_qa_val = jp(config['path_qa'], 'processed', 'valset.pickle')
        with open(path_qa_val, 'rb') as f:
            data_val = pickle.load(f)

        # separate answers into groups according to the biomarker they correspond to (valid only when there is one type of question)
        dict_val_groups = general.group_answers_by_biomarker(answers_best_val_epoch, data_val)
        dict_val_groups['all'] = answers_best_val_epoch['answers'] # add whole answers set to see total plot too

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,10))
        f.suptitle("Val Plots")
        for k, v in dict_val_groups.items():
            # compute metrics
            auc_temp, ap_temp, roc_temp, prc_temp = metrics.compute_roc_prc(v)

            # plot PRC
            ax1.plot(prc_temp[1], prc_temp[0], label = "PRC " + k + ", AP: " + "{:.3f}".format(ap_temp), linewidth=2)

            # plot ROC
            ax2.plot(roc_temp[0], roc_temp[1],label = "ROC " + k + ", AUC: " + "{:.3f}".format(auc_temp), linewidth=2)

        ax1.set_xlabel("recall")
        ax1.set_ylabel("precision")
        ax2.set_xlabel("fpr")
        ax2.set_ylabel("tpr")

        ax1.grid() 
        ax2.grid()   
        ax1.legend()
        ax2.legend()

        plt.savefig(jp(path_logs, 'ROC_PRC_val_each.png'), dpi=300)


        # plot curves for test set, if it has been processed with inference.py
        path_test_answers_file = jp(path_logs, 'answers', 'answers_epoch_0.pt')
        if not os.path.exists(path_test_answers_file):
            raise Exception("Test set answers haven't been generated with inference.py")
        answers_test = torch.load(path_test_answers_file, map_location=torch.device('cpu'))

        auc_test, ap_test, roc_test, prc_test = metrics.compute_roc_prc(answers_test['answers'])
        pf.plot_roc_prc(roc_test, auc_test, prc_test, ap_test, title='Test plots', save=True, path=path_logs, suffix='test')

        # for generation of biomarker-wise metrics, read data
        path_qa_test = jp(config['path_qa'], 'processed', 'testset.pickle')
        with open(path_qa_test, 'rb') as f:
            data_test = pickle.load(f)

        # separate answers into groups according to the biomarker they correspond to (valid only when there is one type of question)
        dict_test_groups = general.group_answers_by_biomarker(answers_test, data_test)
        dict_test_groups['all'] = answers_test['answers'] # add whole answers set to see total plot too

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,10))
        f.suptitle("Test Plots")
        for k, v in dict_test_groups.items():
            # compute metrics
            auc_temp, ap_temp, roc_temp, prc_temp = metrics.compute_roc_prc(v)

            # plot PRC
            ax1.plot(prc_temp[1], prc_temp[0], label = "PRC " + k + ", AP: " + "{:.3f}".format(ap_temp), linewidth=2)
            #ax1.plot([0, 1], [no_skill, no_skill], linestyle='--', color = colors[k], label='No Skill')

            # plot ROC
            ax2.plot(roc_temp[0], roc_temp[1],label = "ROC " + k + ", AUC: " + "{:.3f}".format(auc_temp), linewidth=2)
            #ax2.plot(fpr_dumb, tpr_dumb, linestyle="--", color = "gray", label="No Skill")

        ax1.set_xlabel("recall")
        ax1.set_ylabel("precision")
        ax2.set_xlabel("fpr")
        ax2.set_ylabel("tpr")

        ax1.grid() 
        ax2.grid()   
        #ax1.legend() # exclude legend because too many
        #ax2.legend()

        plt.savefig(jp(path_logs, 'ROC_PRC_test_each.png'), dpi=300)

if __name__ == '__main__':
    main()