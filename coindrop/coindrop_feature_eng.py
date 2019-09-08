'''
Created on Apr 28 2019

@author: enerve
'''

import logging
import math
import torch
from really import util

from really.function import FeatureEng

class CoindropFeatureEng(FeatureEng):
    '''
    Feature engineering for Connect-4
    '''

    def __init__(self,
                 config):
        '''
        Constructor
        '''
        
        # states
        self.num_rows = config.NUM_ROWS
        self.num_cols = config.NUM_COLUMNS
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        self.num_inputs = self.num_rows * self.num_cols
        
        self.teye = torch.eye(self.num_actions()).to(self.device)

    def num_actions(self):
        return self.num_cols
        
    def prefix(self):
        return 'FEv2_'

    def x_adjust(self, B):
        ''' Takes the input params and converts it to an input feature array
        '''

        b = torch.from_numpy(B).to(self.device)
        # Create 2 layers, one for each player, with -1 for empty grids
        empty = (b == 0).float() * -1
        x1 = (b > 0.5).float() + empty
        x2 = (b < -0.5).float() + empty
        return torch.stack([x1, x2], dim=0)
    
    def value_from_output(self, net_output):
        return net_output * 2 -1
    
    def output_for_value(self, value):
        return (value + 1.0) / 2

    def a_index(self, action):
        return action

    def action_from_index(self, a_index):
        return a_index

    def valid_actions_mask(self, B):
        valid_actions_mask = (B[6-1] == 0) * 1
        return torch.from_numpy(valid_actions_mask).to(self.device).float()

    def random_action(self, B):
        V = torch.rand(self.num_cols).to(self.device).float() + 1000 * self.valid_actions_mask(B)
        i = torch.argmax(V).item()
        return i
        
    def prepare_data_for(self, S, a, target):
        hist_x, hist_t, hist_mask = [], [], []
        
        t = self.output_for_value(target)
        for flip in [False, True]:
            if flip:
                S = np.flip(S, axis=1).copy()
                a = 6 - a
            x = self.x_adjust(S)
            m = self.teye[a].clone()
        
            hist_x.append(x)
            hist_t.append(t)
            hist_mask.append(m)
            #                 stS = '%s%d' % (np.array2string(S, separator=''),a)
            #                 if stS in Sdict:
            #                     if Sdict[stS] == 1:
            #                         count_conflict += 1
            #                     Sdict[stS] += 1
            #                 else:
            #                     Sdict[stS] = 1
            #                 if stS in Sdict:
            #                     if Sdict[stS] != t:
            #                         count_conflict += 1
            #                 Sdict[stS] = t
                
        return hist_x, hist_t, hist_mask
    