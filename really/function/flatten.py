'''
Created on 2 May 2019

@author: enerve
'''

import torch.nn as nn
import torch

class Flatten(nn.Module):
    def forward(self, x):
        #N, C, H, W = x.size() # read in N, C, H, W
        #return x.view(N, -1)
        #x = x.view(-1)
        #return x.unsqueeze(0)
        return torch.flatten(x, start_dim=1)

# class ExpandRange(nn.Module):
#     def forward(self, x):
#         return 2 * x - 1