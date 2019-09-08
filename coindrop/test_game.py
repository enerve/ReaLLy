'''
Created on 28 Apr 2019

@author: erwin
'''
import unittest
import numpy as np

from game import Game

class Test(unittest.TestCase):


    def test_has_won(self):
        def trans(Blist):
            Blist.reverse()
            return np.array(Blist)#.T
        
        # Test win condition
        g = Game([])
        B = trans([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])
        assert g.has_won(B, 1, 3, 0) == False
        assert g.has_won(B, 1, 3, 3) == False
        assert g.has_won(B, 1, 6, 5) == False
    
        B = trans([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0]])
        assert g.has_won(B, 1, 3, 0) == True
        assert g.has_won(B, 1, 3, 1) == True
        assert g.has_won(B, 1, 3, 2) == True
        assert g.has_won(B, 1, 3, 3) == True
        assert g.has_won(B, 1, 3, 4) == False
    
        B = trans([[-1,-1,-1,-1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0]])
        assert g.has_won(B, 1, 1, 0) == True
        assert g.has_won(B, 1, 2, 1) == True
        assert g.has_won(B, 1, 3, 2) == True
        assert g.has_won(B, 1, 4, 3) == True
        assert g.has_won(B, 1, 5, 4) == False
        assert g.has_won(B, 1, 3, 1) == False
        assert g.has_won(B, 1, 0, 5) == False
        assert g.has_won(B, -1, 0, 5) == True
        assert g.has_won(B, -1, 1, 5) == True
        assert g.has_won(B, -1, 2, 5) == True
        assert g.has_won(B, -1, 3, 5) == True
        assert g.has_won(B, -1, 4, 5) == False
    
        B = trans([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 1, 1, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0, 0]])
        assert g.has_won(B, 1, 1, 0) == False
        assert g.has_won(B, 1, 2, 1) == False
        assert g.has_won(B, 1, 3, 2) == False
        assert g.has_won(B, 1, 5, 4) == False
        assert g.has_won(B, 1, 3, 1) == False
        assert g.has_won(B, 1, 3, 4) == True
        assert g.has_won(B, 1, 4, 3) == True
        assert g.has_won(B, 1, 5, 2) == True
        assert g.has_won(B, 1, 6, 1) == True

    class TestPlayer():
        def reset_test(self, see_list, move_list):
            self.see_list = see_list
            self.num_sees = 0
            self.move_list = move_list
            self.num_moves = 0
            self.over = False
            
        def init_game(self, initial_state):
            pass
            
        def see_move(self, reward, new_state):
            assert reward == self.see_list[self.num_sees], "see'ing %d at %d" %(reward, self.num_sees)
            self.new_state = new_state

            self.num_sees += 1
            
        def next_move(self):
            a = self.move_list[self.num_moves]
            self.num_moves += 1
            return a
        
        def game_over(self):
            assert self.num_sees == len(self.see_list)
            assert self.num_moves == len(self.move_list)
            self.over = True

    def test_win1(self):
        tp1 = self.TestPlayer()
        tp2 = self.TestPlayer()
        game = Game([tp1, tp2])
        
        tp1.reset_test([0, 0, 0, 0, 0, 0, 1],
                       [0, 1, 2, 3])
        tp2.reset_test([0, 0, 0, 0, 0, 0, -1],
                       [0, 0, 0])
        assert tp1.num_moves == 0
        assert tp2.num_moves == 0
        assert tp1.num_sees == 0
        assert tp2.num_sees == 0
        game.run()
        assert tp1.over == True
        assert tp2.over == True

    def test_win2(self):
        tp1 = self.TestPlayer()
        tp2 = self.TestPlayer()
        game = Game([tp1, tp2])
        
        tp1.reset_test([0, 0, 0, 0, 0, 0, 0, -1],
                       [0, 1, 2, 2])
        tp2.reset_test([0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0])
        game.run()
        assert tp1.over == True
        assert tp2.over == True

    def test_draw(self):
        tp1 = self.TestPlayer()
        tp2 = self.TestPlayer()
        game = Game([tp1, tp2])
        
        
        tp1m = [0, 2, 4, 0, 2, 4, 0, 2, 4, 6, 6]
        tp2m = [1, 3, 5, 1, 3, 5, 1, 3, 5, 6, 6]
        tp1m.extend([1, 3, 5, 1, 3, 5, 1, 3, 5, 6])
        tp2m.extend([0, 2, 4, 0, 2, 4, 0, 2, 4, 6])
        
        tp1.reset_test([0 for _ in range(42)],
                       tp1m)
        tp2.reset_test([0 for _ in range(42)],
                       tp2m)
        game.run()
        assert tp1.over == True
        assert tp2.over == True

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()