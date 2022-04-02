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

from keras.models import Sequential, load_model

from SmartGame.generating.classdef import TickTackToe
from SmartGame.generating.utils.smart_utils import block, block_B, check_if_about, check_if_about_B
from SmartGame.interactive.brains import hacer_y_copiar_desacoplado,processInData
from SmartGame.processing.symmetry_rotation import rotate_to_bottom_left_center_of_mass as rblcm

def AIFriendly_O(TickTackToe,model,scaler, past):
    # This is the symmetry rotation_______________.
    TickTackToe.board = rblcm(TickTackToe.board)[0] #|
    #---------------------------------------------|
    a = TickTackToe.board.ravel().tolist()[0]
    L = TickTackToe.length
    if tuple(a) in past.keys() and tuple(a)!=tuple([-1]*(L**2)):
      return past[tuple(a)], [past]
    pL = TickTackToe.patternLength
    cases = []
    probas = []
    for K1 in range(L):
        for K2 in range(L):
            if TickTackToe.board.item(K1,K2)==-1:
                copia = hacer_y_copiar_desacoplado(TickTackToe,L,pL)
                copia.board.itemset((K1,K2),0)
                b = copia.board.ravel().tolist()[0]
                c = a + b
                c = [int(x) for x in c]
                INPUT = np.asarray(c).reshape(1,-1)          
                result = processInData(model, scaler, INPUT)
                result = result[0][0]
                cases.append(b)
                probas.append(result)
            else: pass
    move = cases[np.argmax(probas)]
    copia = hacer_y_copiar_desacoplado(TickTackToe,L,pL)
    copia.board = np.matrix([x for x in np.asarray(move).reshape((L,L)) ])
    past[tuple(a)] = copia.board
    return copia, [past]      



def AIFriendly_X(TickTackToe,model,scaler, past):
    # This is the symmetry rotation_______________.
    TickTackToe.board = rblcm(TickTackToe.board)[0] #|
    #---------------------------------------------|
    a = TickTackToe.board.ravel().tolist()[0]
    L = TickTackToe.length
    if tuple(a) in past.keys() and tuple(a)!=tuple([-1]*(L**2)):
      print('it already existed')
      return past[tuple(a)], [past]
    pL = TickTackToe.patternLength
    cases = []
    probas = []
    for K1 in range(L):
        for K2 in range(L):
            if TickTackToe.board.item(K1,K2)==-1:
                copia = hacer_y_copiar_desacoplado(TickTackToe,L,pL)
                copia.board.itemset((K1,K2),1)
                b = copia.board.ravel().tolist()[0]
                c = a + b
                c = [int(x) for x in c]
                INPUT = np.asarray(c).reshape(1,-1)          
                result = processInData(model, scaler, INPUT)
                result = result[0][0]
                cases.append(b)
                probas.append(result)
            else: pass
    move = cases[np.argmax(probas)]
    copia = hacer_y_copiar_desacoplado(TickTackToe,L,pL)
    copia.board = np.matrix([x for x in np.asarray(move).reshape((L,L))])
    past[tuple(a)] = copia.board
    return copia, [past]      
