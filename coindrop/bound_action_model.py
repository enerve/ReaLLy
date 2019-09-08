'''
Created on May 13, 2019

@author: enerve
'''

import logging
import torch
from really import util

from really.function import FeatureEng

class BoundActionModel(FeatureEng):
    '''
    Handling Inputs and outputs for Connect-4 based on "bound actions", i.e.,
    the action is first applied to the board state before feeding into the NN,
    and the output is a single node.
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
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        self.num_inputs = self.num_rows * self.num_cols
        
#     def num_actions(self):
#         return self.num_actions
        
    def prefix(self):
        return 'BAM_'

    def feature(self, B):
        ''' Takes the action-bound-state and converts it to an input feature
        '''

        b = torch.from_numpy(B).to(self.device)
        # Create 2 layers, one for each player, with -1 for empty grids
        empty = (b == 0).float() * -1
        x1 = (b > 0.5).float() + empty
        x2 = (b < -0.5).float() + empty
        return torch.stack([x1, x2], dim=0)
