'''
Created on 15 Sep 2019

@author: enerve
'''

import logging

from .exploration_strategy import ExplorationStrategy

class ESPolicy(ExplorationStrategy):
    '''
    An exploration strategy that picks actions stochastically from a learned
    policy approximator.
    '''

    def __init__(self, config, fa, pa):
        '''
        Constructor
        '''
        super().__init__(config, 0, fa)

        self.pa = pa

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
    def prefix(self):
        return "policy"
        
    def pick_action(self, S, moves):
        #action, val, val_list = self.fa.best_action(S)
        action, action_probs = self.pa.pick_action(S)
        
        self.logger.debug("Taking action %s from val list: %s", action,
                          action_probs)
        
        return action
