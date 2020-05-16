'''
Created on 30 Sep 2019

@author: enerve
'''

import logging
import torch
import torch.nn as nn 

from really import util
from really.function import GivenGradient

class PolicyGrader():
    '''
    A grader object will help grade the output of a NN model during training,
    and is responsible for calculating the loss gradient and initiating
    autograd's backward pass.
    '''
        
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        self.criterion = GivenGradient()

        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

    def prefix(self):
        return "policy"

    def loss_stats(self, net_out, w_grads, M):
        with torch.no_grad():
            loss = torch.sum(torch.abs(w_grads).sum(1))
            sum_loss = torch.sum(loss, 0)
            count_loss = torch.sum(loss > 0, 0).float()

        return sum_loss, count_loss
    def compute_gradients(self, net_out, w_grads, M):
        
        outputs, dummy_dW = net_out
        
        criterion_output = self.criterion(outputs, dummy_dW)

        # backward
        criterion_output.backward(w_grads.to(self.device))

        return self.loss_stats(net_out, w_grads, M)