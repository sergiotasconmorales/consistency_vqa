# Project:
#   VQA
# Description:
#   Loss function definitions and getter
# Author: 
#   Sergio Tascon-Morales

from torch import nn
import numpy as np
import torch
from ..models import model_factory
import os

def get_criterion(config, device, ignore_index = None, weights = None):
    # function to return a criterion. By default I set reduction to 'sum' so that batch averages are not performed because I want the average across the whole dataset

    if 'crossentropy' in config['loss']:
        if weights is not None:
            crit = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean', weight=weights).to(device)        
        else:
            crit = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum').to(device)
    elif 'bce' in config['loss']:
        if weights is not None:
            crit = nn.BCEWithLogitsLoss(reduction='mean').to(device)
        else:
            crit = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    else:
        raise ValueError("Unknown loss function.")
    
    if 'mainsub' in config:
        if config['mainsub']:
            if 'squint' in config: # if squint is True, second loss term is MSE
                if config['squint']:
                    mse = nn.MSELoss(reduction='none').to(device)
                    return crit, mse
                else:
                    # create second criterion
                    if config['num_answers'] == 2:
                        ce = nn.BCEWithLogitsLoss(reduction='none').to(device)
                    else:
                        ce = nn.CrossEntropyLoss(reduction='none').to(device)
                    return crit, ce           
            else:
                # create second criterion
                # create second criterion
                if config['num_answers'] == 2:
                    ce = nn.BCEWithLogitsLoss(reduction='none').to(device)
                else:
                    ce = nn.CrossEntropyLoss(reduction='none').to(device)
                return crit, ce
        else:
            return crit
    else:
        return crit


def Q2_score(scores_main, gt_main, scores_sub, gt_sub, suit):
    # create softmax
    sm = nn.Softmax(dim=1)
    probs_main = sm(scores_main)
    _, ans_main = probs_main.max(dim=1) 
    probs_sub = sm(scores_sub)
    _, ans_sub = probs_sub.max(dim=1)

    # Q2 score is implemented to measure the number of Q2 inconsistencies within the batch, meaning in how many cases the model predicted the 
    # main question correctly but failed to answer the associated sub-question correctly. 
    q2_score = torch.sum(torch.logical_and(torch.eq(ans_main, gt_main), torch.logical_not(torch.eq(ans_sub, gt_sub)))*suit)/gt_main.shape[0]

    return q2_score


def fcn1(non_avg_ce, alpha, beta, flag, device):
    # First function (not so good because it's periodic)

    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)

    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
    sub_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(1, len(non_avg_ce), 2)]).to(device))
    # summarize flag vector to be taken in same indexes as main_ce
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))

    func = torch.exp(alpha*sub_ce)*torch.cos(beta*main_ce) - 1 # subtract 1 so that loss term is 0 at (0,0)
    relu = nn.ReLU()
    relued = relu(func)
    return torch.mean(relued[torch.where(flag_reduced>0)])

def fcn2(non_avg_ce, flag, device, gamma=0.5, exp=True):
    # Second function. Based on x(gamma-y)

    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)

    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
    sub_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(1, len(non_avg_ce), 2)]).to(device))
    # summarize flag vector to be taken in same indexes as main_ce
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
    if exp:
        func = torch.exp(sub_ce*(gamma - main_ce))-1
    else:
        func = sub_ce*(gamma - main_ce)
    #torch.exp(alpha*sub_ce)*torch.cos(beta*main_ce) - 1 # subtract 1 so that loss term is 0 at (0,0)
    relu = nn.ReLU()
    relued = relu(func)
    return torch.mean(relued[torch.where(flag_reduced>0)])


def fcn3(non_avg_ce, flag, device, phi = 1, delta=2, exp=True):
    # Inclined plane. Causes decrease in performance for main questions

    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)
    relu = nn.ReLU()
    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
    sub_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(1, len(non_avg_ce), 2)]).to(device))
    # summarize flag vector to be taken in same indexes as main_ce
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
    if exp:
        func = phi*relu(torch.log(relu(sub_ce - delta*main_ce + 1)))
        #func = torch.exp(phi*sub_ce - phi*delta*main_ce)-1
    else:
        func = phi*sub_ce - (phi*delta)*main_ce
    #torch.exp(alpha*sub_ce)*torch.cos(beta*main_ce) - 1 # subtract 1 so that loss term is 0 at (0,0)
    
    relued = relu(func)
    return torch.mean(relued[torch.where(flag_reduced>0)])

def fcn4():
    # dummy function to test baseline
    return

def fcn5(non_avg_ce, flag, device, phi = 1, delta=2, exp=True):
    # "bidirectional version of fcn3"

    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)
    relu = nn.ReLU()
    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
    sub_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(1, len(non_avg_ce), 2)]).to(device))
    # summarize flag vector to be taken in same indexes as main_ce
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
    if exp:
        func1 = phi*relu(torch.log(relu(sub_ce - delta*main_ce + 1)))
        func2 = phi*relu(torch.log(relu(main_ce - delta*sub_ce + 1)))
        #func = torch.exp(phi*sub_ce - phi*delta*main_ce)-1
    else:
        func1 = phi*sub_ce - (phi*delta)*main_ce
        func2 = phi*main_ce - (phi*delta)*sub_ce
    #torch.exp(alpha*sub_ce)*torch.cos(beta*main_ce) - 1 # subtract 1 so that loss term is 0 at (0,0)
    
    relued = relu(func1) + relu(func2)
    return torch.mean(relued[torch.where(flag_reduced>0)])

def fcn7(non_avg_ce, flag, device, phi = 1, delta=2, exp=True):
    # Modified version of fcn3 to correct for Q3 inconsistencies only (decreases main performance)

    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)
    relu = nn.ReLU()
    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
    sub_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(1, len(non_avg_ce), 2)]).to(device))
    # summarize flag vector to be taken in same indexes as main_ce
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
    if exp:
        func = phi*relu(torch.log(relu(main_ce - delta*sub_ce + 1)))
        #func = torch.exp(phi*sub_ce - phi*delta*main_ce)-1
    else:
        func = phi*main_ce - (phi*delta)*sub_ce
    #torch.exp(alpha*sub_ce)*torch.cos(beta*main_ce) - 1 # subtract 1 so that loss term is 0 at (0,0)
    
    relued = relu(func)
    return torch.mean(relued[torch.where(flag_reduced>0)])


def fcn8(non_avg_ce, flag, device, gamma=0.5, exp=True):
    # "bidirectional" fcn2 (doesn't really make sense because changes value of function in the neighborhood of (0,0))

    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)

    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
    sub_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(1, len(non_avg_ce), 2)]).to(device))
    # summarize flag vector to be taken in same indexes as main_ce
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
    if exp:
        func1 = torch.exp(sub_ce*(gamma - main_ce))-1
        func2 = torch.exp(main_ce*(gamma - sub_ce))-1
    else:
        func1 = sub_ce*(gamma - main_ce)
        func2 = main_ce*(gamma - sub_ce)
    #torch.exp(alpha*sub_ce)*torch.cos(beta*main_ce) - 1 # subtract 1 so that loss term is 0 at (0,0)
    relu = nn.ReLU()
    relued = relu(func1) + relu(func2)
    return torch.mean(relued[torch.where(flag_reduced>0)])


def fcn9(non_avg_ce, flag, device, gamma=2.0, exp=True):
    # Alternative function to see if somehow main questions can be improved too. z = gamma*x + beta*y - xy -> Obtained based on the desired gradients (beta set to 1)

    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)

    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
    sub_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(1, len(non_avg_ce), 2)]).to(device))
    # summarize flag vector to be taken in same indexes as main_ce
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
    if exp:
        func = torch.exp(sub_ce*(gamma - main_ce) + 1*main_ce)-1
    else:
        func = sub_ce*(gamma - main_ce) + 1*main_ce
    #torch.exp(alpha*sub_ce)*torch.cos(beta*main_ce) - 1 # subtract 1 so that loss term is 0 at (0,0)
    relu = nn.ReLU()
    relued = relu(func)
    return torch.mean(relued[torch.where(flag_reduced>0)])

def fcn10(non_avg_ce, flag, device, gamma=2.0, exp=True):
    """Dummy
    """
    return 42

class ConsistencyLossTerm(object):
    """Class for consistency loss term with different functions

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __init__(self, config, vocab_words=None):
        if 'consistency_function' not in config:
            return
        else:
            self.fcn = config['consistency_function']
            
            if self.fcn not in globals():
                raise ValueError("Unknown function")
            else:
                self.loss_term_fcn = globals()[self.fcn]

            if 'adaptive' in config:
                self.adaptive = config['adaptive']
                self.min_ce_previous_epoch = np.inf # define first min ce for main-questions as 0 (will be modified after first epoch)
            else:
                self.adaptive = False

            if self.fcn == 'fcn1':
                self.alpha = config['alpha']
                self.beta = config['beta'] # set to value as per config file even though this value won't be used
            elif self.fcn == 'fcn2' or self.fcn == 'fcn8' or self.fcn == 'fcn9':
                self.gamma = config['gamma']
                self.exp = config['exp']
            elif self.fcn == 'fcn3' or self.fcn == 'fcn7':
                self.phi = config['phi']
                self.delta = config['delta']
                self.exp = config['exp']
                self.ces_sub = []
                self.ces_main = []
            elif self.fcn == 'fcn4':
                pass
            elif self.fcn == 'fcn5':
                self.phi = config['phi']
                self.delta = config['delta']
                self.exp = config['exp']
                self.ces_sub = []
                self.ces_main = []
            elif self.fcn == 'fcn10': # if NLP model
                # create model
                self.path_nlp = config['path_nlp']
                self.nlp = model_factory.get_nlp_model(config, vocab_words)
                # update weights with pre-trained model_factory
                try:
                    model_params = torch.load(os.path.join(self.path_nlp, 'best_checkpoint_model.pt'))
                except:
                    model_params = torch.load(os.path.join(self.path_nlp, 'best_checkpoint_model.pt'), map_location=torch.device('cpu'))
                try:
                    self.nlp.load_state_dict(model_params)
                except:
                    self.nlp.module.load_state_dict(model_params)                
            else:
                raise ValueError

    def compute_loss_term(self, non_avg_ce, flag, device):
        # Depending on function, compute loss term accordingly
        if self.fcn == 'fcn1':
            return self.loss_term_fcn(non_avg_ce, self.alpha, self.beta, flag, device)
        elif self.fcn == 'fcn2' or self.fcn == 'fcn8' or self.fcn == 'fcn9':
            return self.loss_term_fcn(non_avg_ce, flag, device, gamma=self.gamma, exp=self.exp)
        elif self.fcn == 'fcn3' or self.fcn == 'fcn7':
            return self.loss_term_fcn(non_avg_ce, flag, device, phi = self.phi, delta=self.delta, exp=self.exp)
        elif self.fcn == 'fcn4':
            return torch.tensor(0)
        elif self.fcn == 'fcn5':
            return self.loss_term_fcn(non_avg_ce, flag, device, phi = self.phi, delta=self.delta, exp=self.exp)
        else:
            raise ValueError('Unknown function. If youre using the NLP model, use compute_loss_term_nlp instead of compute_loss_term')

    def compute_loss_term_nlp(self, q, scores, flag, device):
        """Function to compute the inconsistency score as determined by the NLP model created with fcn10 (See __init__)

        Parameters
        ----------
        q : torch tensor
            batch of questions
        o : torch tensor
            batch of output scores produced by the model (un-softmaxed)
        flag : torch tensor
            flags indicating positions of main questions

        Returns
        -------
        float
            inconsistency score as determined by NLP model
        """

        if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
            return torch.tensor(0)

        # first, obtain the answer indexes for the output scores
        sm = nn.Softmax(dim=1)
        probs = sm(scores) # [B, num_answers]
        _, a = probs.max(dim=1) # [B]

        # separate into even and odd indexes (main vs sub or ind vs ind)
        q_even = torch.index_select(q, 0, torch.tensor([i for i in range(0, len(q), 2)]).to(device)) # [B/2, 23]
        q_odd = torch.index_select(q, 0, torch.tensor([i for i in range(1, len(q), 2)]).to(device)) # [B/2, 23]

        # same for answers
        a_even = torch.index_select(a, 0, torch.tensor([i for i in range(0, len(a), 2)]).to(device)) # [B/2]
        a_odd = torch.index_select(a, 0, torch.tensor([i for i in range(1, len(a), 2)]).to(device)) # [B/2]

        # summarize flag vector to be taken in same indexes as main_ce
        flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device)) # [B/2]

        # use flag_reduced to calculate mq, sq, ma and sa
        mq = q_even[torch.where(flag_reduced.squeeze()>0)] # squeeze is necessary so that whole row is returned
        sq = q_odd[torch.where(flag_reduced.squeeze()>0)] 

        ma = a_even[torch.where(flag_reduced>0)] 
        sa = a_odd[torch.where(flag_reduced>0)]

        sigmoid = nn.Sigmoid()

        return torch.mean(1 - sigmoid(self.nlp(mq, ma.unsqueeze(1), sq, sa.unsqueeze(1)))) # one minus so that it measures inconsistency instead of consitency 

    def log_ces_sub_main(self, non_avg_ce, flag, device):
        # put CE subs and CE main in self.ces_sub and self.ces_main
        if not self.adaptive: # sanity check
            raise ValueError
        main_ce_prev = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
        sub_ce_prev = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(1, len(non_avg_ce), 2)]).to(device))
        flag_reduced_like_main = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
        flag_reduced_like_sub = torch.index_select(flag, 0, torch.tensor([i for i in range(1, len(flag), 2)]).to(device))
        ces_main = main_ce_prev[torch.where(flag_reduced_like_main>0)]
        ces_sub = sub_ce_prev[torch.where(flag_reduced_like_sub>0)]

        self.ces_main.append(torch.mean(ces_main).item())
        self.ces_sub.append(torch.mean(ces_sub).item())

    def update_ce_if_min(self, non_avg_ce, flag, device):
        if not self.adaptive: # sanity check
            raise ValueError
        main_ce_prev = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
        flag_reduced_prev = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
        ces_main = main_ce_prev[torch.where(flag_reduced_prev>0)]
        if ces_main.numel() != 0: # when none of the samples corresponds to a subquestion, beta doesn't matter so let's just use the one in the config file
            min_ce_main = torch.min(ces_main)
            # update parameter in object
        if min_ce_main < self.min_ce_previous_epoch:
            self.min_ce_previous_epoch = min_ce_main

    def update_loss_params(self): 
        if self.fcn == 'fcn1' and self.adaptive: # sanity check
            beta = torch.tensor(np.pi)/self.min_ce_previous_epoch
            beta = beta.clone().detach()
            self.beta = beta # save value so that validation loss is computed with the same function)
            print('New value of beta:', self.beta.item())
        elif self.fcn == 'fcn2' and self.adaptive:
            self.gamma = self.min_ce_previous_epoch.clone().detach()
            print('New value of gamma:', self.gamma.item())
        elif self.fcn == 'fcn3' and self.adaptive:
            # in this case, because we only want to close or open the function, it's enough to change omega only
            self.delta = 2*np.mean(np.array(self.ces_main))/np.mean(np.array(self.ces_sub))
            print('New value of delta:', self.delta)

        self.ces_sub = []
        self.ces_main = []