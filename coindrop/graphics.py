'''
Created on 14 May 2019

@author: enerve
'''

from really import cmd_line
from really import util
import numpy as np
import math

def draw_all(B_list):
    length = len(B_list)
    rows = math.ceil(math.sqrt(length))
    columns = math.floor(math.sqrt(length))

    y1, x1 = 0, 0
    y2, x2 = rows * (6+1) + 1, columns * (7+1) + 1
    G = np.zeros((y2, x2, 3), dtype=np.int)
    G[:, :, 2] = 255
    
    a = b = 0
    for B in B_list:
        place(B, G, a, b)
        a += 1
        if a == columns:
            a = 0
            b += 1
        
    util.heatmap(G, (x1, x2, y1, y2), aspect='equal')

def place(B, G, a, b):
    C = np.zeros((6, 7, 3), dtype=np.int)
    C[:, :, 0] = 255 
    C[:, :, 1] = (B<=0) * 210 + (B==0) * 45
    C[:, :, 2] = (B==0) * 255 
    
    y, x = b * (6+1) + 1, a * (7+1) + 1

    G[y:(y+6), x:(x+7), :] = C

def draw(B):
    C = np.zeros((6, 7, 3), dtype=np.int)
    C[:, :, 0] = 255 
    C[:, :, 1] = (B<=0) * 255 
    C[:, :, 2] = (B==0) * 255 
    
    util.heatmap(C, (0, 7, 0, 6))

def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'TryGraphics'

    def trans(Blist):
        Blist.reverse()
        return np.array(Blist)
    
    B = trans([[-1,-1,-1,-1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 1, 1, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0]])
    
    #draw(B)
    B_list = [B.copy() for i in range(10)]
    draw_all(B_list)    
    
    

if __name__ == '__main__':
    main()