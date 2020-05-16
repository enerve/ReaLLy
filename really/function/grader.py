'''
Created on 30 Sep 2019

@author: enerve
'''

import logging
import torch
import torch.nn as nn 

from really import util

class Grader():
    '''
    A grader object will help grade the output of a NN model during training,
    and is responsible for calculating the loss gradient and initiating
    autograd's backward pass.
    '''
        
    def __init__(self, criterion_str):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.criterion_str = criterion_str
        if criterion_str == 'mse':
            self.criterion = nn.MSELoss(reduce=False)
        elif criterion_str == 'bce':
            self.criterion = nn.BCELoss(reduce=False)
        else:
            self.logger.error("Unkonwn criterion %s", criterion_str)

        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

    def prefix(self):
        return self.criterion_str

    def loss_stats(self, outputs, Y, M):
        Y = Y * M  # b x do
        loss = self.criterion(outputs, Y)  # b x do
        with torch.no_grad():
            # Zero-out the computed losses for the other actions/outputs
            loss *= M   # b x do

            sum_loss = torch.sum(loss, 0)
            count_loss = torch.sum(loss > 0, 0).float()

        return sum_loss, count_loss

    def compute_gradients(self, outputs, Y, M):
        # loss
        Y = Y * M  # b x do
        loss = self.criterion(outputs, Y)  # b x do
        with torch.no_grad():
            # Zero-out the computed losses for the other actions/outputs
            loss *= M   # b x do

        # backward
        loss.backward(torch.ones(loss.shape).to(self.device))

        with torch.no_grad():
            sum_loss = torch.sum(loss, 0)
            count_loss = torch.sum(loss > 0, 0).float()

        return sum_loss, count_loss