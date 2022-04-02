#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""board-visualizers
"""

import os
import pickle
import random
import re
import sys
import time

import pandas as pd
import numpy as np

from classdef import TickTackToe
from math import sqrt

def game_printer(v: tuple):
    """shows an entire game sequentially.
       Specially useful for "data.pkl" keys"""
    for i,x in enumerate(v):
      print(f'hand {i}\n\n')
      board_printer(x)
    return 

def board_printer(tup: tuple):
    """shows board-status at fixed time;
       it won't look pretty in all devices"""
    L = int(sqrt(len(tup)))
    aux = np.asarray(tup).reshape(L,L)
    tabla_visual = [[1 for x in range(L)] for x in range(L)] 
    for x in range(L):
        for y in range(L):
            if aux[x][y]==-1: pass
            elif aux[x][y]==1: tabla_visual[x][y] = 'X'
            elif aux[x][y]==0: tabla_visual[x][y] = 'O'
            else: raise Exception('this was not a valid input!')    
    underline = '_'*5*L + '_.'
    filler = '.'*int(((2+5*L)-len('CURRENT BOARD'))/2)
    print(f'{filler}Current Board{filler}')
    print(underline)
    for x in range(L):
        q = '|'
        for y in range(L):
            if tabla_visual[x][y]==1: q += '|___|'
            elif tabla_visual[x][y]=='X': q += '|_X_|'
            elif tabla_visual[x][y]=='O': q += '|_O_|'
            else: q = '|err|'
        q += '|'        
        print(q)
    print('\n\n\n')
    return

def viewer(n,df):
  """Shows two successive moves. 
     Specially useful for ".csv" rows"""
  board_printer(df.iloc[n,:9])
  board_printer(df.iloc[n,9:-1])
  return

