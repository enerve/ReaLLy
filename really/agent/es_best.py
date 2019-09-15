'''
Created on 12 Sep 2019

@author: enerve
'''

import logging

from .exploration_strategy import ExplorationStrategy

class ESBest(ExplorationStrategy):
    '''
    An exploration strategy that always simply picks the best action
    '''

    def __init__(self, config, fa):
        '''
        Constructor
        '''
        super().__init__(config, 0, fa)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
    def prefix(self):
        return "best"
        
    def pick_action(self, S, moves):
        action, val, val_list = self.fa.best_action(S)
        
        self.logger.debug("Taking action %s from val list: %s", action,
                          val_list)
        
        return action
