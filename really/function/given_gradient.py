'''
Created on 23 Sep 2019

@author: enerve
'''

import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.autograd import Variable

class Given(Function):
    '''
        A custom autograd function that outputs a dummy variable in forward pass,
        andsimply passes through the given gradient in the backward pass.
    '''

    @staticmethod
    def forward(ctx, inp, dummy_dW):
        ctx.save_for_backward(inp, dummy_dW)

        output = torch.ones(dummy_dW.shape) # shape of future w_grads
        return output
        
    @staticmethod
    def backward(ctx, w_grads):
        inp, dummy_dW = ctx.saved_tensors
                
        d_inp = torch.ones(inp.shape)
        
        # Ignore loss or output and blindly return given gradient
        
        return d_inp, w_grads


class GivenGradient(nn.Module):
    '''
    A loss-function module that simply passes through the given gradient in the
    backward pass.
    '''

    def forward(self, X, dummy_dW):
        return Given.apply(X, dummy_dW)
        
        
if __name__ == '__main__':
    
    # Test GivenGradient and its function Given.
    
    from torch.autograd import Variable
    import random
    import collections
    
    seed = 123
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Test LinearForward net with GivenGradient as criterion
    from really.function import AllSequential
    from really.function.linear_forward_given_gradient import LinearForward

    net = AllSequential(collections.OrderedDict([
        ('lfgg', LinearForward(10)),
        ]))
    criterion = GivenGradient()
    
    
    batchsize = 5
    input =  Variable(torch.Tensor(batchsize, 10).uniform_(-1, 1))
    input.requires_grad = True
    w_grads = Variable(torch.Tensor(batchsize, 10).uniform_(-1, 1))
    w_grads.requires_grad = False

    #params = net.parameters()
    net_out, _ = net(input)    
    outputs, dummy_dW = net_out
    
    loss = criterion(outputs, dummy_dW)
    
    loss.backward(w_grads)
    
    for p in net.parameters():
        print("Param: %s" % p)
        print("Grad: %s" % p.grad)
         
    print("X grad: ", input.grad)
    