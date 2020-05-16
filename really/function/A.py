'''
Created on 23 Sep 2019

@author: enerve
'''

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from really.function.given import Given

class A(Function):

    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X, W, b)
        print("(A) forward")

        outp = X.mm(W.t())#.squeeze()
        #outp = torch.ones(5, 1) #TODO: remove
        dummy_dW = torch.ones(5, 10)
        #dummy_dW = X
        
        return (outp, dummy_dW)#(out, W)
        
    @staticmethod
    def backward(ctx, dummy_dO, w_grads):
        X, W, b = ctx.saved_tensors
        print("(A) dummy_dO: ", dummy_dO)
        print("(A) w_grads: ", w_grads)
        
        dX = X * w_grads / (W + 1e-7)
        dW = w_grads.sum(0, keepdim=True)
        db = torch.zeros(1)
        
        print("(A) dW set to ", dW)
        
        return dX, dW, db

class AM(nn.Module):

    def __init__(self, input_features):
        super(AM, self).__init__()
        
        self.weights = nn.Parameter(torch.Tensor(1, input_features))
        self.bias = nn.Parameter(torch.Tensor(1))

        self.weights.data.uniform_(-0.1, 0.1)
        self.bias.data.uniform_(-0.1, 0.1)
        
        print("Init weights to", self.weights)

    def forward(self, X):
        return A.apply(X, self.weights, self.bias)
        
        
    
# if __name__ == '__main__':
#     import numpy as np
#     
#     def relu(x):
#         return np.maximum(0,x)
#     
#     def reluDerivative(x):
#         #x = x.copy()
#         x[x<=0] = 0
#         x[x>0] = 1
#         return x
#   
#     training_inputs = np.array([[9, 0 , 1],
#         [7, 1, 1],
#         [8, 0, 1],
#         [5, 1, 1]
#         ])
#     
#     training_outputs = np.array([[9, 7, 8, 5]]).T
#     
#     np.random.seed(10)
#     
#     synaptic_weights = 2 * np.random.random((3,1)) - 1
#     
#     
#     for iteration in range(100000):
#     
#         outputs = relu(np.dot(training_inputs, synaptic_weights))
#     
#     
#         error = training_outputs - outputs
#         adjustments = error * reluDerivative(outputs)
#         synaptic_weights += np.dot(training_inputs.T, adjustments )
#     
#     print("output after training: \n" , outputs)