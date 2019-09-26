'''
Created on 24 Sep 2019

@author: enerve
'''

import torch
from torch.autograd.function import Function
from torch.autograd import Variable

class Given(Function):
    '''
        A custom autgrad function that calculates a Linear output 
        in the forward phase but uses the given precomputed gradient 
        in the backward phase
    '''

    @staticmethod
    def forward(ctx, X, W, b, w_grads, b_grad):
        ctx.save_for_backward(X, W, b, w_grads, b_grad)

        out = Variable(X.mm(W.t()))
        return out
        
    @staticmethod
    def backward(ctx, dO):
        X, W, b, w_grads, b_grad = ctx.saved_tensors
        
        # Calculate dX, given that we know w_grads. Essentially, we're
        # reverse-engineering from the new output=X*(W+dW)
        # X=5,10  W=1,10  w_grads=1,10
        dX = X * w_grads / (W + 1e-7)
        # dW needs to sum/average across the batch
        dW = w_grads.sum(0, keepdim=True)
        db = b_grad
        
        print("done backward")
        return dX, dW, db, None
        