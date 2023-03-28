# Project:
#   VQA
# Description:
#   Inference for single image
# Author: 
#   Sergio Tascon-Morales

import os
import yaml
from os.path import join as jp
import comet_ml
import numpy as np
import random
import torch 
import pickle
import torch.nn as nn 
import misc.io as io
from PIL import Image
from misc import printer
from core.datasets import loaders_factory, visual, nlp
from core.models import model_factory
from core.train_vault import criterions, optimizers, train_utils, looper, comet

# read config name from CLI argument --path_config
args = io.get_config_file_name(single=True)

def main():
    # read config file
    config = io.read_config(args.path_config)
    config['train_from'] = 'best' # set this parameter to best so that best model is loaded for validation part
    config['comet_ml'] = False

    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    if device.type == 'cpu':
        config['num_workers'] = 1
        config['pin_memory'] = False
        config['data_parallel'] = False 
        config['cuda'] = False

    # now I have to generate dataloader with single sample
    #train_loader, vocab_words, vocab_answers, index_unk_answer = loaders_factory.get_vqa_loader('train', config)

    # read vocabs
    path_processed = jp(config['path_qa'], 'processed')
    path_map_index_word = jp(path_processed, 'map_index_word.pickle')
    path_map_index_answer = jp(path_processed, 'map_index_answer.pickle')

    with open(path_map_index_word, 'rb') as f:
                vocab_words = pickle.load(f)
    with open(path_map_index_answer, 'rb') as f:
                vocab_answers = pickle.load(f)

    # Read and preprocess image
    path_image = args.path_image
    img = Image.open(path_image).convert('RGB')
    tr = visual.default_transform(config['size'])
    img = tr(img).unsqueeze_(0)

    # Create binary image the size of the image (as mask for questions about whole image)
    mask = torch.from_numpy(np.array(Image.open(args.path_mask).convert('L'))).to(torch.float32).unsqueeze_(0).unsqueeze_(0)
    if torch.max(mask) >1: # normalize if necessary
        mask = mask/torch.max(mask)
    #mask = torch.ones(1,1, config['size'], config['size'], dtype=torch.float32)

    # Process question: tokenize, generate vector
    question = args.question
    tokens = nlp.tokenize_single_question(config['tokenizer'], question)
    tokens_UNK = nlp.add_UNK_token_single(tokens, list(vocab_words.values()))
    encoded_question = nlp.encode_single_question(tokens_UNK, {w:i for i,w in vocab_words.items()}, config['max_question_length'])
    question_sample = torch.LongTensor(encoded_question).unsqueeze_(0)

    # create model
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    # create optimizer
    optimizer = optimizers.get_optimizer(config, model)


    # load weights from best epoch
    best_epoch, _, _, _, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)

    model.eval()

    sm = nn.Softmax(dim=1)

    with torch.no_grad():
        output = model(img, question_sample, mask)
        if 'squint' in config: # squint returns output and att maps
            if config['squint']:
                output = output[0]
        probs = sm(output)
        _, pred = probs.max(dim=1)

    answer_text = vocab_answers[pred]

    # print answer
    print('Summary')
    print('Image path:', path_image)
    print('Mask path:', args.path_mask)
    print('Question:', question)
    print('Answer:', answer_text)

if __name__ == '__main__':
    main()