'''
Created on Apr 28, 2019

@author: enerve
'''

import logging
import random

from really.agent import Agent

class RandomAgent(Agent):
    '''
    
    '''

    def __init__(self):
        ''' Initialize for a new game, and note player's start state '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.recent_R = 0
        self.recent_played = 0
        
    def prefix(self):
        pref = "random"
        return pref

    # ---------------- Single game ---------------

    def init_game(self, initial_state, initial_heights):
        self.S = initial_state
        self.total_R = 0
        
    def see_outcome(self, reward, new_state, h, moves=0):
        ''' Observe the effects on this agent of an action taken - possibly by
            another agent.
            reward Reward earned by this agent
        '''
        self.S = new_state
        self.total_R += reward
         
    def next_move(self):
        ''' Agent's turn. Chooses the next move '''

        options = []
        for x in range(7):
            if self.S[5, x] == 0:
                options.append(x)
                
        idx = random.randint(0, len(options)-1)
        return options[idx]

    def collect_stats(self, ep, num_episodes):
        if (ep+1)% 100 == 0:
            #self.logger.debug("RandomAgent recent R: %d/%d" % (self.recent_R, self.recent_played))
            self.recent_R = 0
            self.recent_played = 0
        
        self.recent_R += self.total_R
        self.recent_played += 1 
