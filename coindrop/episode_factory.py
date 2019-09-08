'''
Created on Sep 1, 2019

@author: enerve
'''

from really.episode_factory import EpisodeFactory as EF
from .game import Game

class EpisodeFactory(EF):

    def new_episode(self, explorer_list):
        return Game(explorer_list)