# Project:
#   VQA
# Description:
#   Metrics computation
# Author: 
#   Sergio Tascon-Morales

import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

def vqa_accuracy(predicted, true):
    """ Compute the accuracies for a batch according to VQA challenge accuracy"""
    # in this case true is a [B, 10] matrix where ever row contains all answers for the particular question
    _, predicted_index = predicted.max(dim=1, keepdim=True) # should be [B, 1] where every row is an index
    agreement = torch.eq(predicted_index.view(true.size(0),1), true).sum(dim=1) # row-wise check of times that answer in predicted_index is in true

    return torch.min(agreement*0.3, torch.ones_like(agreement)).float().sum() # returning batch sum


def batch_strict_accuracy(predicted, true):
    # in this case true is a [B] tensor with the answers 
    sm = nn.Softmax(dim=1)
    probs = sm(predicted)
    _, predicted_index = probs.max(dim=1) # should be [B, 1] where every row is an index
    return torch.eq(predicted_index, true).sum() # returning sum

def batch_binary_accuracy(predicted, true):
    # input predicted already contains the indexes of the answers
    return torch.eq(predicted, true).sum() # returning sum

def compute_auc_ap(targets_and_preds):
    # input is an Nx2 tensor where the first column contains the target answer for all samples and the second column containes the sigmoided predictions
    targets_and_preds_np = targets_and_preds.cpu().numpy()
    auc = roc_auc_score(targets_and_preds_np[:,0], targets_and_preds_np[:,1]) # eventually take np.ones((targets_and_preds_np.shape[0],)) - targets_and_preds_np[:,1]
    ap = average_precision_score(targets_and_preds_np[:,0], targets_and_preds_np[:,1], pos_label=1)
    return auc, ap

def compute_roc_prc(targets_and_preds, positive_label = 1):
    y_true = targets_and_preds[:,0]
    y_pred = targets_and_preds[:,1]
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred, pos_label=positive_label)
    ap = average_precision_score(y_true, y_pred, pos_label=positive_label)
    return auc, ap, (fpr, tpr, thresholds_roc), (precision, recall, thresholds_pr)

def consistency_introspect(qa_pairs, pivot=None):
    """Computes consistency for VQA-Introspect dataset.

    Parameters
    ----------
    qa_pairs : _type_
        List of dictionaries with information about the predictions, including the prediction of the model

    pivot : int
        To accelerate the processing of sub-questions, pivot is the number of non-sub questions (i.e. main+ind)
    """
    # should be easier than for DME because there are no sub-categories 
    #* step 1: List all correct main questions
    main_correct = [e['question_id'] for e in qa_pairs if e['pred']==e['answer_index']]

    #* step 2: List sub-questions for correct main questions
    sub_correct = [e['pred']==e['answer_index'] for e in tqdm(qa_pairs[pivot:]) if e['parent'] in main_correct]

    #* step 3: Return consistency metric comptued from number of correct sub-questions (number of True's) divided by length
    return np.sum(np.array(sub_correct))/len(sub_correct)


def consistency(qa_pairs_test):
    """Function to compute consistency metric as defined in Selvaraju et al. 2020 (ratio between correctly answered sub-questions for images
    for which main question (grade question) was answered correctly.)

    Parameters
    ----------
    qa_pairs_test : list
        List of dictionaries after adding the prediction for every QA pair

    Returns
    -------
    float
        Consistency score
    """

    # GRADE 0
    total_sub_correct = 0 # counter for total number of correct sub-questions for correct main_questions with any grade
    total_sub = 0 # counter for total number of sub-questions for correct main-questions with any grade
    # list all samples for which model correctly answered main (grade) questions and grade is 0.
    correct_zero_samples_main = [e for e in qa_pairs_test if e['question_type']=='grade' and e['answer']==e['pred'] and e['answer']==0]
    # list images for which grade was correctly assigned
    correct_zero_images_main = list(set([e['image_name'] for e in correct_zero_samples_main]))
    # now for every correct image, find out how many correct relevant sub-questions there are. I have to do it separately for fovea&whole and inside because
    # not all inside questions imply incosistencies. 
    for img in correct_zero_images_main:
        # whole, fovea and inside (in this case all inside questions constitute an inconsistency, if answered wrongly)
        zero_samples_sub_all = [e for e in qa_pairs_test if e['image_name'] == img and e['question_type'] != 'grade']
        zero_samples_sub_all_correct = [e for e in zero_samples_sub_all if e['answer'] == e['pred']]
        total_sub_correct += len(zero_samples_sub_all_correct)
        total_sub += len(zero_samples_sub_all)
        
    # GRADE 1
    correct_one_samples_main = [e for e in qa_pairs_test if e['question_type']=='grade' and e['answer']==e['pred'] and e['answer']==1]
    correct_one_images_main = list(set([e['image_name'] for e in correct_one_samples_main]))
    for img in correct_one_images_main:
        # whole, fovea
        one_samples_sub_fovea_whole = [e for e in qa_pairs_test if e['image_name'] == img and (e['question_type'] == 'fovea' or e['question_type'] == 'whole')]
        one_samples_sub_fovea_whole_correct = [e for e in one_samples_sub_fovea_whole if e['answer'] == e['pred']]
        total_sub_correct += len(one_samples_sub_fovea_whole_correct)
        total_sub += len(one_samples_sub_fovea_whole)

        # inside (only those centered at macula)
        one_samples_sub_inside = [e for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'inside' and e['center'] == 'macula']
        one_samples_sub_inside_correct = [e for e in one_samples_sub_inside if e['answer'] == e['pred']]
        total_sub_correct += len(one_samples_sub_inside_correct)
        total_sub += len(one_samples_sub_inside)

    # GRADE 2
    correct_two_samples_main = [e for e in qa_pairs_test if e['question_type']=='grade' and e['answer']==e['pred'] and e['answer']==2]
    correct_two_images_main = list(set([e['image_name'] for e in correct_two_samples_main]))
    for img in correct_two_images_main:
        # whole, fovea
        two_samples_sub_fovea_whole = [e for e in qa_pairs_test if e['image_name'] == img and (e['question_type'] == 'fovea' or e['question_type'] == 'whole')]
        two_samples_sub_fovea_whole_correct = [e for e in two_samples_sub_fovea_whole if e['answer'] == e['pred']]
        total_sub_correct += len(two_samples_sub_fovea_whole_correct)
        total_sub += len(two_samples_sub_fovea_whole)

        # inside (only those centered at macula)
        two_samples_sub_inside = [e for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'inside' and e['center'] == 'macula']
        two_samples_sub_inside_correct = [e for e in two_samples_sub_inside if e['answer'] == e['pred']]
        total_sub_correct += len(two_samples_sub_inside_correct)
        total_sub += len(two_samples_sub_inside)

    if total_sub == 0:
        consistency_score = 0
    else:
        consistency_score = total_sub_correct/total_sub
    return consistency_score

def consistencies_q2_q3(qa_pairs_test):
    # first, compute q2 consistency using old function
    q2_consistency = 100*consistency(qa_pairs_test)

    # now compute two Q3 consistencies: A loose one and a strict one. The loose one follows this definition: How often the model predicts the main-question correctly,
    # given that it answered a subquestion correctly. The strict score only considers cases in which all subquestions were answered correctly for a particular image.

    total_main_correct = 0 # counter for total number of correct sub-questions for correct main_questions with any grade
    total_main = 0 # counter for total number of sub-questions for correct main-questions with any grade

    # loose Q3 consistency
    # 'whole' questions first
    correct_whole = [e for e in qa_pairs_test if e['question_type']=='whole' and e['answer']==e['pred']]
    correct_whole_images_sub = list(set([e['image_name'] for e in correct_whole]))

    for img in correct_whole_images_sub:
        # whole, fovea and inside (in this case all inside questions constitute an inconsistency, if answered wrongly)
        main_all = [e for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'grade'] # should have single element in this case
        assert len(main_all) == 1
        main_correct = [e for e in main_all if e['answer'] == e['pred']]
        total_main_correct += len(main_correct)
        total_main += len(main_all)

    # same for 'fovea' questions
    correct_fovea = [e for e in qa_pairs_test if e['question_type']=='fovea' and e['answer']==e['pred']]
    correct_fovea_images_sub = list(set([e['image_name'] for e in correct_fovea]))

    for img in correct_fovea_images_sub:
        # whole, fovea and inside (in this case all inside questions constitute an inconsistency, if answered wrongly)
        main_all = [e for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'grade'] # should have single element in this case
        assert len(main_all) == 1
        main_correct = [e for e in main_all if e['answer'] == e['pred']]
        total_main_correct += len(main_correct)
        total_main += len(main_all)

    # now inside questions 
    correct_inside = [e for e in qa_pairs_test if e['question_type']=='inside' and e['center'] == 'macula']
    correct_inside_images_sub = list(set([e['image_name'] for e in correct_inside]))

    for img in correct_inside_images_sub:
        # whole, fovea and inside (in this case all inside questions constitute an inconsistency, if answered wrongly)
        main_all = [e for e in qa_pairs_test if e['image_name'] == img and e['question_type'] == 'grade'] # should have single element in this case
        assert len(main_all) == 1
        main_correct = [e for e in main_all if e['answer'] == e['pred']]
        total_main_correct += len(main_correct)
        total_main += len(main_all)

    q3_consistency_loose = 100*(total_main_correct/total_main)


    # strict Q3 consistency
    # I need to find, for every image, those for which all available sub-questions (3 or 2) are correct. The length of that will be the denominator. The numerator
    # will be the number of correct grades (main questions) for those cases
    possible = 0
    possible_correct = 0
    all_images = list(set([e['image_name'] for e in qa_pairs_test]))
    for img in all_images:
        sub_curr_image = [e for e in qa_pairs_test if e['image_name'] == img and e['role'] == 'sub']
        sub_correct = [e for e in sub_curr_image if e['answer'] == e['pred']]
        if len(sub_curr_image) == len(sub_correct): # If all available sub-questions were correct, check main
            possible += 1
            main_curr_image = [e for e in qa_pairs_test if e['image_name'] == img and e['answer'] == e['pred'] and e['question_type'] == 'grade']
            possible_correct += len(main_curr_image)

    if possible==0:
        q3_consistency_strict = 0
    else:
        q3_consistency_strict = 100*(possible_correct/possible)

    return q2_consistency, q3_consistency_loose, q3_consistency_strict