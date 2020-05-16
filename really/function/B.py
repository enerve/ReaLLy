'''
Created on 23 Sep 2019

@author: enerve
'''

import torch
import torch.nn as nn
#from really.function.given import Given
from torch.autograd.function import Function
from torch.autograd import Variable

class B(Function):

    @staticmethod
    def forward(ctx, inp, dummy_dW):#, w_grads):
        #ctx.save_for_backward(inp, dummyDW)

        #output = X.t() * w_grads.t() # dummy
        #output = dummyDW
        output = torch.ones(5, 10)
        return output
        
    @staticmethod
    def backward(ctx, w_grads):
        #inp, dummyDW = ctx.saved_tensors
        #dX = None
        print("(B) w_grads: ", w_grads)
        
        #dummyDW = torch.ones(dummyDW.shape)
        
        #print("(B) dummyDW: ", dummyDW)
        
        dummy_dO = torch.ones(5, 1)#None
        
        return dummy_dO, w_grads #, None


class BM(nn.Module):

    def forward(self, X, dummy_dW):
        return B.apply(X, dummy_dW)
        
        
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

    from really.function.A import AM

    net = AM(10)
    criterion = BM()
    
    
    batchsize = 5
    input =  Variable(torch.Tensor(batchsize, 10).uniform_(-1, 1))
    input.requires_grad = True
    w_grads = Variable(torch.Tensor(batchsize, 10).uniform_(-1, 1))
    w_grads.requires_grad = False

    net_out = net(input)    
    outputs, dummy_dW  = net_out
    
    loss = criterion(outputs, dummy_dW)
    
    #loss.backward(torch.ones(10, batchsize))
    loss.backward(w_grads)

    for p in net.parameters():
        print("----")
        print("Param: %s" % p)
        print("Grad: %s" % p.grad)
         
    print("X grad: ", input.grad)
    