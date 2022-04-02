#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""core.py 's utilities
"""

import os
import pickle
import random
import re
import sys
import time

import pandas as pd
import numpy as np

from SmartGame.generating.classdef import TickTackToe

def saver(dic,flag):
    a = os.listdir()
    a = [x for x in a if x[-4:]=='.pkl']
    pattern = re.compile(f'results([0-9]+).pkl')
    indice = ''
    if a:
        for x in a:
            try:
                q = int(re.search(pattern,x).group(1))
                if not indice: indice=0
                if q>= indice: indice = q + 1
            except: pass    
    print('saving with flag: ',flag)
    with open(f'./../data/results{flag}.pkl','wb') as w:
        pickle.dump(dic, w)    
    return



def watch(t):
    L = t.length
    B = copy_m(t)
    B = np.asarray([[str(y) for y in x] for x in B])
    for x in range(L):
        for y in range(L):
            if B[x][y]==str(-1): B[x][y]= ' '
            elif B[x][y]==str(1): B[x][y]= 'O'
            else: B[x][y]= 'X'
    print(B)
    return 

def copy(A):
    L = A.length
    pL = A.patternLength
    B = TaTeTi(L,pL)
    for x in range(L):
        for y in range(L):
            B.board.itemset((x,y),A.board.item(x,y))
    return B

def copy_m(A):
    L = A.length
    B = np.asarray([[-1 for x in range(L)] for y in range(L)])
    for x in range(L):
        for y in range(L):
            B[x][y] = A.board.item(x,y)
    return B



def aux1(log,extra):
    if extra==0:
        return log
    if extra==1:
        return log[0:2]    
    if extra==2:
        return log[2:4]
    if extra==3:
        return log[4:6]
    if extra==4:
        if len(log)==8: return log[-2:]
        else: raise
    else: return log

def aux2(log,extra): 
    if extra==0:
        return log[:-1]
    if extra==1:
        return log[0:2]
    if extra==2:
        return log[2:4]
    if extra==3:
        if len(log)<6: raise IndexError("it's ok: ended before 3 could be answered! (optimized for speed)")
        else: return log[4:6]
    if extra==4:
        if len(log)==9: return log[-3:-1]
        else: raise IndexError("it's ok: ended before 4 could be answered! (optimized for speed)")
    else: return log


def if_verbose(verbose: bool, t, time:int):
  if verbose:
    watch(t)
    time.sleep(time)
  return
