'''
Created on Sep 1, 2019

@author: enerve
'''

from really.episode_factory import EpisodeFactory as EF
from .episode import Episode

class EpisodeFactory(EF):

    def __init__(self, track, car, num_junctures):
        self.track = track
        self.car = car
        self.num_junctures = num_junctures

    def new_episode(self, explorer_list):
        driver = explorer_list[0]
        return Episode(driver, self.track, self.car, self.num_junctures)