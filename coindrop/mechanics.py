'''
Created on 1 May 2019

@author: enerve
'''

def has_won(B, coin, x, y):
    ''' Checks if 'coin' has won the game by playing at x,y '''
    
    if _count(B, coin, x, y, 0, 1) >= 4:
        return True
    if _count(B, coin, x, y, 1, 0) >= 4:
        return True
    if _count(B, coin, x, y, 1, 1) >= 4:
        return True
    if _count(B, coin, x, y, 1, -1) >= 4:
        return True
    
    return False
    
def _count(B, coin, x, y, a, b):
    ''' Counts consecutive coins of 'coin' around x,y along direction a,b
    '''
    
    count = 0
    x_, y_ = x, y
    for i in range(4):
        if B[y_, x_] != coin:
            break
        count += 1
        x_ += a
        if not 0 <= x_ < 7:
            break 
        y_ += b
        if not 0 <= y_ < 6:
            break 
        
    if count == 0:
        return 0
        
    a, b = -a, -b
    x_, y_ = x+a, y+b
    if not 0 <= x_ < 7 or not 0 <= y_ < 6:
        return count
    rem = 4 - count 
    for i in range(rem):
        if B[y_, x_] != coin:
            break
        count += 1
        x_ += a
        if not 0 <= x_ < 7:
            break 
        y_ += b
        if not 0 <= y_ < 6:
            break 
        
    return count

