#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""non-random players are defined
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
from SmartGame.generating.utils.smart_utils import block, block_B, check_if_about, check_if_about_B

def smart_O(t, proba: float):
    # IF the coin is below the randomness threshold THEN random move
    if np.random.rand()<=proba: 
         t.movesO()
         return t

    L = t.length

    # IF there is a winning-move THEN do it
    t, status = check_if_about(t)
    if status: return t

    # IF the oponent can win in one move THEN block
    ( (ind_1, ind_2), resultatio) = block(t)
    if resultatio: 
        t.board.itemset((ind_1,ind_2),0)
        return t

    # IF none of the above THEN attempt to build a pattern
    (i,j),estado = process_dict(otra(t))
    if estado: 
        t.board.itemset((i,j),0)
        return t

    # IF you are here then a TIE is inminent
    t.movesO()
    return t


def smart_X(t, proba : float):
    # IF the coin is below the randomness threshold THEN random move
    if np.random.rand()<=proba: 
         t.movesX()
         return t

    L = t.length

    # IF empty board THEN random initialization  
    if (t.board==np.matrix([[-1]*L for x in range(L) ]) ).all(): 
        t.movesX()
        return t

    # IF there is a winning-move THEN do it
    t, status = check_if_about_B(t)
    if status: return t

    # IF the oponent can win in one move THEN block
    ( (ind_1, ind_2), resultatio) = block_B(t)
    if resultatio: 
        t.board.itemset((ind_1,ind_2),1)
        return t

    # IF none of the above THEN attempt to build a pattern
    (i,j),estado = process_dict_B(otra_B(t))
    if estado: 
        t.board.itemset((i,j),1)
        return t

    # IF you are here then a TIE is inminent
    t.movesX()
    return t


def h(x,y,k,q,L):
    a = x
    b = y-k+q 
    if 0<=a<=L and 0<=b<=L: return a,b
    else: raise IndexError

def v(x,y,k,q,L):
    a = x-k+q
    b = y 
    if 0<=a<=L and 0<=b<=L: return a,b
    else: raise IndexError

def dne(x,y,k,q,L):
    a = x+k-q
    b = y-k+q 
    if 0<=a<=L and 0<=b<=L: return a,b
    else: raise IndexError

def dnw(x,y,k,q,L):
    a = x-k+q
    b = y-k+q 
    if 0<=a<=L and 0<=b<=L: return a,b
    else: raise IndexError
  
def otra(t,FL=0):
    L = t.length
    msg = 'cant rewrite values on the board jeje'
    pL = t.patternLength
    data = {}
    for x in range(0,pL+1):
        data[x] = []
    cases = ['h','v','dnw','dne']
    info = {'h':h, 'v':v, 'dne':dne, 'dnw':dnw}
    for x in range(L):
        for y in range(L):
            if t.board.item(x,y) in [FL, -1] : 
                for name in cases:
                    for k in range(pL):
                        try:
                            counterG = 0
                            indices = []
                            for q in range(pL): 
                                a,b = info[name](x,y,k,q,L)
                                temp = t.board.item(a,b)
                                if temp==FL: 
                                    counterG += 1
                                    indices.append((a,b,True))
                                elif temp==-1: 
                                    indices.append((a,b,False))
                                else:
                                    raise Exception(msg)
                            shorten_code = data[counterG]
                            shorten_code += [(x,y,k,name,tuple(indices))]
                            data[counterG] = shorten_code 
                        except: pass
            else: pass 
    return data



def otra_B(t,FL=1):
    L = t.length
    msg = 'cant rewrite values on the board jeje'
    pL = t.patternLength
    data = {}
    for x in range(0,pL+1):
        data[x] = []
    cases = ['h','v','dnw','dne']
    info = {'h':h, 'v':v, 'dne':dne, 'dnw':dnw}
    for x in range(L):
        for y in range(L):
            if t.board.item(x,y) in [FL, -1] : 
                for name in cases:
                    for k in range(pL):
                        try:
                            counterG = 0
                            indices = []
                            for q in range(pL): 
                                a,b = info[name](x,y,k,q,L)
                                temp = t.board.item(a,b)
                                if temp==FL: 
                                    counterG += 1
                                    indices.append((a,b,True))
                                elif temp==-1: 
                                    indices.append((a,b,False))
                                else:
                                    raise Exception(msg)
                            shorten_code = data[counterG]
                            shorten_code += [(x,y,k,name,tuple(indices))]
                            data[counterG] = shorten_code 
                        except: pass
            else: pass 
    return data


def process_dict(dicc):
    moves = {}
    if not all(list(dicc.values())): 
        return (0,0), False
    for y in dicc.keys():
        if not dicc[y]: pass
        else:
            moves[y] =  tuple([x[4] for x in dicc[y]])
            moves[y] = tuple( [x[0:2] for x in sum(moves[y],()) if not x[2] ])
    try:
        best = moves[max(moves.keys())]
    except Exception as ins:                   
        print(dicc[1])
        sys.exit(1)
    ranker = {}
    for x in best: ranker[x] =0
    for x in best: ranker[x] += 1
    jugadas = [x for x in ranker.keys() if ranker[x]==max(ranker.values())]
    jugada = random.choice(jugadas)
    return jugada, True     
 

def process_dict_B(dicc):
    moves = {}
    if not all(list(dicc.values())): 
        return (0,0), False
    for y in dicc.keys():
        if not dicc[y]: pass
        else:
            moves[y] =  tuple([x[4] for x in dicc[y]])
            moves[y] = tuple( [x[0:2] for x in sum(moves[y],()) if not x[2] ])
    try:
        best = moves[max(moves.keys())]
    except Exception as ins:         
        print('Unexpected error at "process_dict_B": code', ins.args)
        print('(exiting because it isn\' clear if it\'s serious)')
        sys.exit(1)
    ranker = {}
    for x in best: ranker[x] =0
    for x in best: ranker[x] += 1
    jugadas = [x for x in ranker.keys() if ranker[x]==max(ranker.values())]
    jugada = random.choice(jugadas)
    return jugada, True     
