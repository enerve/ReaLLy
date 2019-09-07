'''
Created on 30 Apr 2019

@author: enerve
'''

import numpy as np
import random
from .exploration_strategy import ExplorationStrategy

class ESDepth(ExplorationStrategy):
    '''
    classdocs
    '''

    def __init__(self, config, explorate, fa):
        '''
        Constructor
        '''
        super().__init__(config, explorate, fa)

        self.N = np.zeros(config.NUM_COLUMNS * config.NUM_ROWS)
        
    def prefix(self):
        return "depth"

    def pick_action(self, S, moves):
        n = self.N[moves]
        N0 = self.explorate
        epsilon = N0 / (N0 + n)
      
        if random.random() >= epsilon:
            # Pick best
            action = self.fa.best_action(S)[0]
        else:
            action = self.fa.random_action(S)
        
        self.N[moves] += 1

        return action
