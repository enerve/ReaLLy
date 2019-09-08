'''
Created on May 22, 2019

@author: enerve
'''

import logging
import torch
from really import util

from really.function import FeatureEng

class FEElevation(FeatureEng):
    '''
    Feature engineering that has two layers of the baord state, one for each player,
    and where each empty slot is number in a way to represent elevation from
    playing space: E.g. -10 for immediately playable spots, -9 for one higher,
    etc. 
    '''

    def __init__(self,
                 config):
        '''
        Constructor
        '''
        
        # states
        self.num_rows = config.NUM_ROWS
        self.num_cols = config.NUM_COLUMNS
        # actions
        self.num_actions = config.NUM_COLUMNS
        
        #self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        self.num_inputs = self.num_rows * self.num_cols
        
    def prefix(self):
        return 'FEelv_'

    def feature(self, B):
        ''' Takes the action-bound-state and converts it to an input feature
        '''

        b = torch.from_numpy(B)
        
        N = torch.Tensor(list(range(6)))
        N = torch.ger(N, torch.ones(7))
        E = (b==0).float()
        R = E.sum(dim=0)

        empty = E * (N + R - 15)
        
        # Create 2 layers, one for each player, with -1 for empty grids
        x1 = (b > 0.5).float() + empty
        x2 = (b < -0.5).float() + empty
        return torch.stack([x1, x2], dim=0)
