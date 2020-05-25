'''
Created on 23 Sep 2019

@author: enerve
'''

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from really import util

DEVICE = None

class LinearForwardGivenGradient(Function):
    '''
        A custom autograd function that calculates a linear output 
        in the forward phase along with a dummy weight variable, and in the backward
        pass takes in a dummy gradient for the linear output and a precomputed 
        weight gradient to be used directly.
    '''


    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X, W, b)

        outp = X.mm(W.t())
        dummy_dW = torch.ones(X.shape).to(DEVICE) # TODO: can we precompute this?
        
        return (outp, dummy_dW)
        
    @staticmethod
    def backward(ctx, dummy_dO, w_grads):
        X, W, b = ctx.saved_tensors
        
        # Calculate dX, given that we know w_grads. Essentially, we're
        # reverse-engineering from the new output=X*(W+dW)
        # X=5,10  W=1,10  w_grads=1,10
        dX = X * w_grads / (W + 1e-7)
        # dW needs to sum/average across the batch
        dW = w_grads.sum(0, keepdim=True)
        db = torch.zeros(1).to(DEVICE)
        
        #print("(A) dW set to ", dW)
        
        return dX, dW, db

class LinearForward(nn.Module):
    '''
    A module that calculates a Linear output in the forward phase 
    but uses the given precomputed gradient in the backward phase
    '''


    def __init__(self, input_features):
        super(LinearForward, self).__init__()

        self.weights = nn.Parameter(torch.Tensor(1, input_features))
        self.bias = nn.Parameter(torch.Tensor(1))

        # TODO: Do better weight initialization
        self.weights.data.uniform_(-0.1, 0.1)
        self.bias.data.uniform_(-0.1, 0.1)

        global DEVICE
        DEVICE = torch.device('cuda' if util.use_gpu else 'cpu')

    def forward(self, X):
        return LinearForwardGivenGradient.apply(X, self.weights, self.bias)
