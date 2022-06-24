# Project:
#   VQA
# Description:
#   Script to check how often questions from VQA2 mention the available segmented categories (or their synonyms)
# Author: 
#   Sergio Tascon-Morales

import json
import yaml
import re
from tqdm import tqdm
from os.path import join as jp

# path to categories (i.e. to file that contains dictionaries with key:value pairs in the form [int] image_id: [list] categories)
path_categories = '/home/sergio814/Documents/PhD/code/data/coco'
file_categories = 'map_img_id_categories_<s>.json'
file_synonyms = 'syn.yaml'

path_vqa2 = '/home/sergio814/Documents/PhD/code/data/VQA2/qa'
file_vqa2 = 'v2_OpenEnded_mscoco_<s>2014_questions.json'

# auxiliary functions to tokenize
def clean_text(text):
    text = text.lower().replace("\n", " ").replace("\r", " ")
    # replace numbers and punctuation with space
    punc_list = '!"#$%&()*+,-./:;<=>?@[\]^_{|}~' + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.translate(t)

    # replace single quote with empty character
    t = str.maketrans(dict.fromkeys("'`", ""))
    text = text.translate(t)

    return text

def tokenizer_nltk(text, tokenizer):
    text = clean_text(text)
    tokens = tokenizer(text)
    return tokens

def tokenizer_spacy(text, tokenizer):
    text = clean_text(text)
    tokens = list(tokenizer(text))
    tokens_list_of_strings = [str(token) for token in tokens]
    return tokens_list_of_strings

def tokenizer_re(text):
    WORD = re.compile(r'\w+')
    text = clean_text(text)
    tokens = WORD.findall(text)
    return tokens

def tokenize_single_question(tokenizer_name, question_text):

    if tokenizer_name == 'nltk':
        from nltk import word_tokenize
    elif tokenizer_name == 'spacy':
        from spacy.tokenizer import Tokenizer
        from spacy.lang.en import English
        lang = English()
        tokenizer = Tokenizer(lang.vocab)

    if tokenizer_name == 'nltk':
        tokens = tokenizer_nltk(question_text, word_tokenize)
    elif tokenizer_name == 'spacy':
        tokens = tokenizer_spacy(question_text, tokenizer)
    elif tokenizer_name == 're':
        tokens = tokenizer_re(question_text)
    else:
        raise ValueError('Unknown tokenizer')
    return tokens


s = 'val' # put for loop later



# read vqa2 questions
with open(jp(path_vqa2, file_vqa2.replace('<s>', s))) as f:
    vqa2 = json.load(f)['questions']

# read image_id - category map
with open(jp(path_categories, file_categories.replace('<s>', s))) as f:
    cats = json.load(f)

# read synonyms
with open(jp(path_categories, file_synonyms)) as f:
    syn = yaml.load(f, Loader=yaml.FullLoader)

cnt_total = 0
collected = []
for q in tqdm(vqa2):
    # get image id
    img_id = q['image_id']
    question = q['question']
    tokens = tokenize_single_question('re', question)
    # get categories for which segmentations are available for curr image
    if str(img_id) not in cats:
        continue
    categories = cats[str(img_id)]
    # now check if category (or synonyms) is mentioned in the question. If at least one of the categories is mentioned, increase counter
    cnt_mentioned = 0
    for c in categories:
        # TODO if multiple words in category, check every one
        if c in tokens or c+'s' in tokens:
            cnt_mentioned += 1
            continue
        if c in syn:
            # also try c
            # try synonyms
            for c_ in syn[c]:
                if c_ in tokens or c_+'s' in tokens:
                    cnt_mentioned += 1
                    break # no need to try other synonyms
            # if multiple words, try each of them too
            #if len(c.split(' '))>1: 
            #     for w in c.split(' '):
            #         if w in question:
            #            cnt_mentioned += 1
            #            break

    if cnt_mentioned >= 1: # if at least one of the categories is mentioned in question
        cnt_total += 1
        collected.append(q)
print('Percentage mentioned (at least one category): ', cnt_total, '/', len(vqa2),  'or', "{:.2f}%".format(100*cnt_total/len(vqa2)))
a = 42
