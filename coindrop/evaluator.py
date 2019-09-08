'''
Created on 23 May 2019

@author: enerve
'''

import logging

from really.evaluator import Evaluator as Eval

from .game import Game

class Evaluator(Eval):
    
    def __init__(self, test_agent, opponent, num_runs):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.test_agent = test_agent
        self.opponent = opponent
        self.num_runs = num_runs
    
    def evaluate(self, ep):
        # Test performance in actual games
        for start_first in [True, False]:
            self.logger.debug("-------- testing as %s player ----------",
                              "first" if start_first else "second")
            agent_sum_score = 0
            num_wins = num_losses = 0
            agent_sum_win_moves = 0
            agent_sum_lose_moves = 0
            if start_first:
                players = [self.test_agent, self.opponent] 
            else: 
                players = [self.opponent, self.test_agent]
            for tep in range(self.num_runs):
                game = Game(players)
                game.run()
                # increased significance for a win / loss the sooner it happens
                factor = (42-self.test_agent.moves)/42
                score = factor * self.test_agent.G
                agent_sum_score += score
                if score > 0:
                    num_wins += 1 
                    agent_sum_win_moves += self.test_agent.moves
                elif score < 0:
                    num_losses += 1
                    agent_sum_lose_moves += self.test_agent.moves
            self.logger.debug("FA agent R: %0.2f   (%d vs %d) / %d" % 
                              (agent_sum_score, num_wins, num_losses, self.num_runs))                
            if num_wins > 0:
                self.logger.debug("Avg #moves for win: %0.2f" % (agent_sum_win_moves/num_wins))
            if num_losses > 0:
                self.logger.debug("Avg #moves for loss: %0.2f" % (agent_sum_lose_moves/num_losses))

                
    def report_stats(self, pref=""):
        pass