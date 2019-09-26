'''
Created on 23 Sep 2019

@author: enerve
'''

import torch
import torch.nn as nn
from really.function.given import Given

class GivenGradient(nn.Module):
    '''
    A module that calculates a Linear output in the forward phase 
    but uses the given precomputed gradient in the backward phase
    '''


    def __init__(self, input_features):
        '''
        Constructor
        '''
        super(GivenGradient, self).__init__()
        
        self.weights = nn.Parameter(torch.Tensor(1, input_features))
        self.bias = nn.Parameter(torch.Tensor(1))

        self.weights.requires_grad = True
        self.bias.requires_grad = True

        # TODO: Do better weight initialization
        self.weights.data.uniform_(-0.1, 0.1)
        self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, X, G):
        return Given.apply(X, self.weights, self.bias, G)
        
        
        
if __name__ == '__main__':
    
    # Test GivenGradient and its function Given.
    
    from torch.autograd import Variable
    import random
    
    seed = 123
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Test a 10-node output layer
    gg = GivenGradient(10)
    
    batchsize = 5
    input =  Variable(torch.Tensor(batchsize, 10).uniform_(-1, 1))
    input.requires_grad = True
    w_grads = Variable(torch.Tensor(batchsize, 10).uniform_(-1, 1))
    w_grads.requires_grad = False
    bias_grad = torch.zeros(1)
    print("Given grads sum: %s" % w_grads.sum(0, keepdim=True))
    
    net = gg
    #params = net.parameters()    
    output = net(input, w_grads, bias_grad)
    
    output.backward(torch.ones(5, 1))
    
    for p in net.parameters():
        print("Param: %s" % p)
        print("Grad: %s" % p.grad)
    print("dX: ", input.grad)
    
    print(output)
    