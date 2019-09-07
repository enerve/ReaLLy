'''
Created on 14 Feb 2019

@author: erwin
'''

class FeatureEng(object):
    '''
    Base class for engineered features input vectors
    '''

    def x_adjust(self):
        pass

    def valid_actions_mask(self, B):
        pass

    def random_action(self, B):
        pass
