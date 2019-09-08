'''
Created on Apr 28, 2019

@author: enerve
'''

from coindrop import mechanics
from really import util

import logging
import numpy as np
import random

from really.agent import Agent

class LookaheadAgent(Agent):
    '''
    
    '''

    def __init__(self, depth):
        ''' Initialize for a new game, and note player's start state '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.depth = depth
        
        self.episodes_history = []
        self.test_episodes_history = []

        self.recent_R = 0
        self.recent_played = 0
        self.sum_moves = 0
        
    def prefix(self):
        pref = "look%d" % self.depth
        return pref

    # ---------------- Single game ---------------

    def init_game(self, initial_state, initial_heights):
        self.S = initial_state
        self.R = 0
        self.h = initial_heights
        self.G = 0
        self.steps_history = []
        
    def see_outcome(self, reward, new_state, new_heights, moves=None):
        ''' Observe the effects on this agent of an action taken - possibly by
            another agent.
            reward Reward earned by this agent
        '''
        self.S = new_state
        self.h = new_heights
        self.R += reward
        self.G += reward
        if moves: self.moves = moves
         
    def next_move(self):
        ''' Agent's turn. Chooses the next move '''

        _, options = self._best_move(self.depth, coin=1, return_options=True)
        # choose randomly from best options
        idx = random.randint(0, len(options)-1)
        A = options[idx]
        
        self.steps_history.append((self.R, self.S, A))

        self.R = 0  # prepare to collect rewards

        return A

    def _best_move(self, depth, coin, return_options=False):
        ''' Returns score of my (coin's) best next move '''

        if depth == 0:
            # can't lookahead this deep
            return 0, None
            
        # Look at all my options and return my best move value
    
        best = None
        options = [] if return_options else None
        for x in range(7):
            y = self.h[x]
            if y < 6:
                self.S[y, x] = coin
                self.h[x] += 1
                if mechanics.has_won(self.S, coin, x, y):
                    val = 100
                else:
                    # check how good the opponent's response might be
                    b, _ = self._best_move(depth-1, -coin)
                    val = -b/1.2
                if best is None or val > best:
                    best = val
                    if return_options: options = [x]
                elif val == best:
                    if return_options: options.append(x)

                self.h[x] -= 1
                self.S[y, x] = 0
                
        if best is None:
            # board is full
            best = 0

        return best, options

    def game_over(self):
        ''' Wrap up game  '''
        self.steps_history.append((self.R, None, None))

        self.sum_moves += self.moves

    def save_episode_for_training(self):
        self.episodes_history.append(self.steps_history)

    def save_episode_for_testing(self):
        self.test_episodes_history.append(self.steps_history)

    def collect_stats(self, ep, num_episodes):
        if (ep+1)% 100 == 0:
            #self.logger.debug("RandomAgent recent R: %d/%d" % (self.recent_R, self.recent_played))
            self.recent_R = 0
            self.recent_played = 0
        
        self.recent_R += self.G
        self.recent_played += 1 

    def total_reward(self):
        
        # increased significance of a win or a loss the sooner it happened
        factor = (42-self.moves)/42
        
        return factor * self.G

    def get_episodes_history(self):
        return self.episodes_history

    def get_test_episodes_history(self):
        return self.test_episodes_history

    def decimate_history(self):
        self.episodes_history = [self.episodes_history[i] for i in
                                 range(len(self.episodes_history)) 
                                 if i % 10 > 0]
        self.test_episodes_history = [self.test_episodes_history[i] for i in 
                                      range(len(self.test_episodes_history)) 
                                      if i % 10 > 0]
    def store_episode_history(self, fname):
        EH = np.asarray(self.episodes_history)
        util.dump(EH, fname, "EH")
        VEH = np.asarray(self.test_episodes_history)
        util.dump(VEH, fname, "VEH")
 
    def load_episode_history(self, fname, subdir):
        self.episodes_history = [[(s[0], s[1], s[2]) for s in sh]
                                 for sh in util.load(fname, subdir, suffix="EH")]
        self.test_episodes_history = [[(s[0], s[1], s[2]) for s in sh] 
                                      for sh in util.load(fname, subdir, suffix="VEH")]
