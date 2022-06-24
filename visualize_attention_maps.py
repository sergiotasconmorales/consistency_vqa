# Project:
#   VQA
# Description:
#   Script to visualize attention maps
# Author: 
#   Sergio Tascon-Morales


import os 
import yaml 
from os.path import join as jp
import comet_ml 
import torch 
import torch.nn as nn 
import misc.io as io
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from core.datasets import loaders_factory
from core.datasets.visual import default_inverse_transform as dit
from core.models import model_factory
from misc import dirs
from core.train_vault import criterions, optimizers, train_utils, looper, comet

torch.manual_seed(1234) # use same seed for reproducibility

# read config name from CLI argument --path_config
args = io.get_config_file_name()

# define hook for storing attention maps
att = {}
def get_att_map(name):
    def hook(model, input, output):
        att[name] = output.detach()
    return hook 

def get_question_text(vocab_words, indexes):
    q = ''
    for idx in indexes:
        if idx != 0:
            q = q + ' ' + vocab_words[idx]
        else:
            q += '?'
            break
    return q


def main():
    # read config file
    config = io.read_config(args.path_config)

    config['train_from'] = 'best' # set this parameter to best so that best model is loaded for validation part
    config['comet_ml'] = False

    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    # load data
    train_loader, vocab_words, vocab_answers, index_unk_answer = loaders_factory.get_vqa_loader('train', config)
    
    test_loader = loaders_factory.get_vqa_loader('test', config)

    # create model
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    # create criterion
    criterion = criterions.get_criterion(config, device, ignore_index = index_unk_answer)

    # create optimizer
    optimizer = optimizers.get_optimizer(config, model)

    best_epoch, _, _, _, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)

    dirs.create_folder(jp(path_logs, 'att_maps'))

    # now iterate trough test loader 
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            print('Batch', i+1, '/', len(test_loader))
            # move data to device
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            mask_a = sample['maskA'].to(device)

            if config['model'] == 'VQARS_1':
                model.attention_mechanism.conv2.register_forward_hook(get_att_map('attention_mechanism.conv2'))
                output = model(visual, question, mask_a) 
                m = nn.Sigmoid()
                pred = (m(output.data.cpu())>0.5).to(torch.int64)
                k = att['attention_mechanism.conv2']
                h = k.clone()
                h = h.view(5,2,14*14)
                sm = nn.Softmax(dim=2)
                h_out = sm(h)
                g_out = h_out.view(5,2,14,14)
                for i_s in range(g_out.shape[0]): # for every element of the batch
                    image = dit()(visual[i_s]).permute(1,2,0).numpy()
                    f, ax = plt.subplots(1, 3)
                    f.tight_layout()
                    ax[0].imshow(image)
                    ax[0].axis('off')
                    ax[0].set_title(vocab_words[question[i_s,2]] + ", R: " + str(answer[i_s].item()))
                    if pred[i_s].item() == answer[i_s].item():
                        f.set_facecolor("green")
                    else:
                        f.set_facecolor("r")
                    for i_glimpse in range(g_out.shape[1]): # for every glimpse
                        img1 = g_out[i_s, i_glimpse, :, :].numpy()
                        heatmap = cv2.resize(img1, (image.shape[1], image.shape[0]))
                        heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                        heatmap = np.uint8(255*heatmap)
                        #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        norm = plt.Normalize()
                        heatmap = plt.cm.jet(norm(heatmap))
                        superimposed = heatmap[:,:,:3] * 0.4 + image*mask_a[i_s].permute(1,2,0).numpy()
                        superimposed = 255*(superimposed - np.min(superimposed))/(np.max(superimposed) - np.min(superimposed))
                        ax[i_glimpse+1].imshow(superimposed.astype(np.uint8))
                        ax[i_glimpse+1].axis('off')
                        ax[i_glimpse+1].set_title("Glimpse " + str(i_glimpse+1))
                        #plt.imshow(superimposed.astype(np.uint8)) 
                        #plt.title("q_id: " + str(question_indexes[i_s].item()) + ", glimpse:" + str(i_glimpse))
                        #plt.show()

                    #plt.suptitle("q_id: " + str(question_indexes[i_s].item()))
                    plt.savefig(jp(path_logs, 'att_maps', str(question_indexes[i_s].item()) + '.png') ,bbox_inches='tight')
                    plt.close()
            elif config['model'] == 'VQARS_4':
                model.attention_mechanism.conv2.register_forward_hook(get_att_map('attention_mechanism.conv2'))
                output = model(visual, question, mask_a) 
                m = nn.Sigmoid()
                pred = (m(output.data.cpu())>0.5).to(torch.int64)
                k = att['attention_mechanism.conv2']
                h = k.clone()
                h = h.view(5,2,14*14)
                sm = nn.Softmax(dim=2)
                h_out = sm(h)
                g_out = h_out.view(5,2,14,14)
                for i_s in range(g_out.shape[0]): # for every element of the batch
                    image = dit()(visual[i_s]).permute(1,2,0).numpy()
                    f, ax = plt.subplots(1, 3)
                    f.tight_layout()
                    ax[0].imshow(image)
                    # show mask on original image
                    masked = np.ma.masked_where(mask_a[i_s].permute(1,2,0).numpy() ==0, mask_a[i_s].permute(1,2,0).numpy())
                    ax[0].imshow(masked, 'jet', interpolation='none', alpha=0.5)
                    ax[0].axis('off')
                    ax[0].set_title(vocab_words[question[i_s,2]] + ", R: " + str(answer[i_s].item()))
                    if pred[i_s].item() == answer[i_s].item():
                        f.set_facecolor("green")
                    else:
                        f.set_facecolor("r")
                    for i_glimpse in range(g_out.shape[1]): # for every glimpse
                        img1 = g_out[i_s, i_glimpse, :, :].numpy()
                        heatmap = cv2.resize(img1, (image.shape[1], image.shape[0]))
                        heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                        heatmap = np.uint8(255*heatmap)
                        #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        norm = plt.Normalize()
                        heatmap = plt.cm.jet(norm(heatmap))
                        superimposed = heatmap[:,:,:3] * 0.4 + image # mask_a[i_s].permute(1,2,0).numpy()
                        superimposed = 255*(superimposed - np.min(superimposed))/(np.max(superimposed) - np.min(superimposed))
                        ax[i_glimpse+1].imshow(superimposed.astype(np.uint8))
                        ax[i_glimpse+1].axis('off')
                        ax[i_glimpse+1].set_title("Glimpse " + str(i_glimpse+1))
                    plt.savefig(jp(path_logs, 'att_maps', str(question_indexes[i_s].item()) + '.png') ,bbox_inches='tight')
                    plt.close()
            elif config['model'] == 'VQARS_6':
                model.attention_mechanism.conv2.register_forward_hook(get_att_map('attention_mechanism.conv2'))
                output = model(visual, question, mask_a) 
                m = nn.Sigmoid()
                pred = (m(output.data.cpu())>0.5).to(torch.int64)
                k = att['attention_mechanism.conv2']
                h = k.clone()
                h = h.view(5,2,14*14)
                sm = nn.Softmax(dim=2)
                h_out = sm(h)
                g_out = h_out.view(5,2,14,14)
                for i_s in range(g_out.shape[0]): # for every element of the batch
                    image = dit()(visual[i_s]).permute(1,2,0).numpy()
                    f, ax = plt.subplots(1, 3)
                    f.tight_layout()
                    ax[0].imshow(image)
                    # show mask on original image
                    masked = np.ma.masked_where(mask_a[i_s].permute(1,2,0).numpy() ==0, mask_a[i_s].permute(1,2,0).numpy())
                    ax[0].imshow(masked, 'jet', interpolation='none', alpha=0.5)
                    ax[0].axis('off')
                    ax[0].set_title(vocab_words[question[i_s,2]] + ", R: " + str(answer[i_s].item()))
                    if pred[i_s].item() == answer[i_s].item():
                        f.set_facecolor("green")
                    else:
                        f.set_facecolor("r")
                    for i_glimpse in range(g_out.shape[1]): # for every glimpse
                        img1 = g_out[i_s, i_glimpse, :, :].numpy()
                        heatmap = cv2.resize(img1, (image.shape[1], image.shape[0]))
                        heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                        heatmap = np.uint8(255*heatmap)
                        #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        norm = plt.Normalize()
                        heatmap = plt.cm.jet(norm(heatmap))
                        superimposed = heatmap[:,:,:3] * 0.4 + image # mask_a[i_s].permute(1,2,0).numpy()
                        superimposed = 255*(superimposed - np.min(superimposed))/(np.max(superimposed) - np.min(superimposed))
                        ax[i_glimpse+1].imshow(superimposed.astype(np.uint8))
                        ax[i_glimpse+1].axis('off')
                        ax[i_glimpse+1].set_title("Glimpse " + str(i_glimpse+1))
                    plt.savefig(jp(path_logs, 'att_maps', str(question_indexes[i_s].item()) + '.png') ,bbox_inches='tight')
                    plt.close()
            elif config['model'] == 'VQARS_7':
                model.attention_mechanism.conv2.register_forward_hook(get_att_map('attention_mechanism.conv2'))
                output = model(visual, question, mask_a) 
                sm = nn.Softmax(dim=1)
                probs = sm(output)
                _, pred = probs.max(dim=1)
                k = att['attention_mechanism.conv2']
                h = k.clone()
                h = h.view(5,2,14*14)
                sm = nn.Softmax(dim=2)
                h_out = sm(h)
                g_out = h_out.view(5,2,14,14)
                for i_s in range(g_out.shape[0]): # for every element of the batch

                    li = list(question[i_s].numpy())
                    if 19 not in li: # if not fovea question
                        continue
                    image = dit()(visual[i_s]).permute(1,2,0).numpy()
                    f, ax = plt.subplots(1, 3, figsize=(50,50))
                    f.tight_layout()
                    ax[0].imshow(image)
                    # show mask on original image
                    if not np.count_nonzero(mask_a[i_s].numpy()) == mask_a[i_s].shape[-1]*mask_a[i_s].shape[-2]:
                        masked = np.ma.masked_where(mask_a[i_s].permute(1,2,0).numpy() ==0, mask_a[i_s].permute(1,2,0).numpy())
                        ax[0].imshow(masked, 'jet', interpolation='none', alpha=0.5)
                    ax[0].axis('off')
                    ax[0].set_title(get_question_text(vocab_words, question[i_s].numpy()) + "\n GT: " + str(vocab_answers[answer[i_s].item()]) + ", Pred: " + str(vocab_answers[pred[i_s].item()]) )
                    if pred[i_s].item() == answer[i_s].item():
                        f.set_facecolor("green")
                    else:
                        f.set_facecolor("r")
                    for i_glimpse in range(g_out.shape[1]): # for every glimpse
                        img1 = g_out[i_s, i_glimpse, :, :].numpy()
                        heatmap = cv2.resize(img1, (image.shape[1], image.shape[0]))
                        heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                        heatmap = np.uint8(255*heatmap)
                        #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        norm = plt.Normalize()
                        heatmap = plt.cm.jet(norm(heatmap))
                        superimposed = heatmap[:,:,:3] * 0.4 + image # mask_a[i_s].permute(1,2,0).numpy()
                        superimposed = 255*(superimposed - np.min(superimposed))/(np.max(superimposed) - np.min(superimposed))
                        ax[i_glimpse+1].imshow(superimposed.astype(np.uint8))
                        ax[i_glimpse+1].axis('off')
                        ax[i_glimpse+1].set_title("Glimpse " + str(i_glimpse+1))
                    plt.savefig(jp(path_logs, 'att_maps', str(question_indexes[i_s].item()) + '.png') ,bbox_inches='tight')
                    plt.close()
            else:
                raise NotImplementedError

    # decide which functions are used for training depending on number of possible answers (binary or not)
    _, validate = looper.get_looper_functions(config)

    # infer 
    metrics, results = validate(test_loader, model, criterion, device, 0, config, None) 
    print("Test set was evaluated for epoch", best_epoch-1, "which was the epoch with the highest", config['metric_to_monitor'], "during training")
    print(metrics)
    train_utils.save_results(results, 0, config, path_logs) # test results saved as epoch 0

    # produce results for the train data too, for the best epoch
    metrics_train, results_train = validate(train_loader, model, criterion, device, 1000, config, None)
    print("Metrics after inference on the train set, best epoch")
    print(metrics_train)
    train_utils.save_results(results_train, 1000, config, path_logs)

    # produce results for the train data, for the last epoch
    config['train_from'] = 'last'
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    # create criterion
    criterion = criterions.get_criterion(config, device, ignore_index = index_unk_answer)

    # create optimizer
    optimizer = optimizers.get_optimizer(config, model)

    # update model's weights
    best_epoch, _, _, _, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)
    metrics_train, results_train = validate(train_loader, model, criterion, device, 1001, config, None)
    print("Metrics after inference on the train set, last epoch")
    print(metrics_train)
    train_utils.save_results(results_train, 1001, config, path_logs)

if __name__ == '__main__':
    main()