'''
Created on 30 Apr 2019

@author: enerve
'''

import numpy as np
import random
from .exploration_strategy import ExplorationStrategy

class ESLayers(ExplorationStrategy):
    '''
    classdocs
    '''

    def __init__(self, config, explorate, fa):
        '''
        Constructor
        '''
        super().__init__(config, explorate, fa)
        self.num_rows = config.NUM_ROWS

        self.N = np.zeros((config.NUM_ROWS, 3, 3, 3, 3, 3, 3, 3))
        
        self.recent_epsilon = 1.0
        
    def prefix(self):
        return "lay"
        
    def pick_action(self, S, moves):
        N0 = self.explorate

        m = 1.0
        for lay in range(self.num_rows):
            Nlay = self.N[lay]
            n = Nlay[tuple(S[lay] + 1)]
            m *= N0 / (N0 + n)
            Nlay[tuple(S[lay] + 1)] += 1
        
        epsilon = m
        
        if moves >= 6:
            f = 0.99
            self.recent_epsilon = f * self.recent_epsilon + (1-f) * epsilon 
      
        if random.random() >= epsilon:
            # Pick best
            action = self.fa.best_action(S)[0]
        else:
            action = self.fa.random_action(S)
        
        return action
