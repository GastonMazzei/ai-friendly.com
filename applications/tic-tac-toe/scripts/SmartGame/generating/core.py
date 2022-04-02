#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""core-dispatcher:
   recieves the request to generate
   N games under specs, handles it and
   finally calls the results-saver
"""

import os
import pickle
import random
import re
import sys
import time

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import load_model

from SmartGame.generating.classdef import TickTackToe
from SmartGame.generating.utils.core_utils import saver
from SmartGame.generating.smart import smart_O, smart_X
from SmartGame.generating.enhaced import AIFriendly_O, AIFriendly_X


def core(Ngames : int, L : int, pL : int, verbose: bool=False, enhace=False, perspective='', **kwargs): 
    """
    kwargs:

        random_o = 0-1  <-- probability of the "O-move" being random
        random_x = 0-1  <-- probability of the "X-move" being random

        alt_o = 'algo' (OR 'ai') <--             ' ' 
        alt_x = 'algo' (OR 'ai') <-- alternative to random: encoded pattern-seeking
                                                          algorithm or ai-network 
    """

    random_O = kwargs.get('random_o',1)
    random_X = kwargs.get('random_x',1)
    alt_O = kwargs.get('alt_o','algo')
    alt_X = kwargs.get('alt_x','algo')

    results = {}
    

    if enhace:
      # comment this and they will play against random-|
      perspective = 'both'                            #|
      #------------------------------------------------|
      if perspective=='x':
        scaler_X = StandardScaler().fit(pd.read_csv(f'../data/processed-x.csv').to_numpy()[:,:-1])
        try:
          model_X = load_model(f'./../data/models/model-x_enhace')
          print('using previous enhaced models for enhacement! (recursive training)')
        except:
          model_X = load_model(f'./../data/models/model-x')
        X_mover = (lambda x,y: AIFriendly_X(x, model_X, scaler_X,y) )
        O_mover = (lambda x,y: (smart_O(x, random_O),[y]))  
      elif perspective=='o':
        scaler_O = StandardScaler().fit(pd.read_csv(f'../data/processed-o.csv').to_numpy()[:,:-1])
        try:
          model_O = load_model(f'./../data/models/model-o_enhace')
          print('using previous enhaced models for enhacement! (recursive training)')
        except:
          model_O = load_model(f'./../data/models/model-o')
        O_mover = (lambda x,y: AIFriendly_O(x, model_O, scaler_O,y) )
        X_mover = (lambda x,y: (smart_X(x, random_X),[y]))  
      else:
        scaler_X = StandardScaler().fit(pd.read_csv(f'../data/processed-x.csv').to_numpy()[:,:-1])
        scaler_O = StandardScaler().fit(pd.read_csv(f'../data/processed-o.csv').to_numpy()[:,:-1])
        try:
          model_X = load_model(f'./../data/models/model-x_enhace')
          print('using previous enhaced models for enhacement! (recursive training)')
        except:
          model_X = load_model(f'./../data/models/model-x')
        X_mover = (lambda x,y: AIFriendly_X(x, model_X, scaler_X,y) )
        try:
          model_O = load_model(f'./../data/models/model-o_enhace')
          print('using previous enhaced models for enhacement! (recursive training)')
        except:
          model_O = load_model(f'./../data/models/model-o')
        O_mover = (lambda x,y: AIFriendly_O(x, model_O, scaler_O,y) )
    else: 
      X_mover = (lambda x,y: (smart_X(x, random_X),[y]))  
      O_mover = (lambda x,y: (smart_O(x, random_O),[y]))  

    past = [{}]
    for x in range(Ngames): 
        try:
            # (0) Report
            #
            if x%1000==0: 
              print(f'Lap N{x}')
              if False:
                # for "enhace"-debugging 
                print(f'past dictionary has {len(past[0].keys())} keys')
                threshold = 20
                if len(past[0].keys())<threshold: 
                  #for k,v in past[0].values(): print(k,v,sep=' ',end='\t')
                  print(past)
            #
            # (1) Initialize
            #
            log = []
            t = TickTackToe(L,pL)
            i = 0
            #
            # (2) Play
            #
            while i==0: 
                temp = []
                try:
                    # (2.0)   "X moves"

                    if enhace and np.random.rand()<=random_X:
                      t.movesX()
                    else:
                      t,past = X_mover(t,past[0])

                    temp.append(tuple(t.board.ravel().tolist()[0]))
             
                    # (2.1)   "if X won, break"
                    if t.checkX(): 
                        i = 1
                        log += temp
                        break    

                    # (2.2)   "O moves"
                    if enhace and np.random.rand()<=random_O:
                      t.movesO()
                    else:
                      t,past = O_mover(t,past[0])

                    temp.append(tuple(t.board.ravel().tolist()[0]))
                    log += temp

                    # (2.3)   "if O won, break"
                    if t.checkO():
                        i = 3    
                        break

                except:
                    #    "if an exception occurred, it was raised by a tie"
                    log += temp
                    
                    i = 2
                    break

 
            # (3) Convert "list" to "tuple"
            #     -"tuples" are hashable; 
            #            "lists" are not-
            #
            log = tuple(log)

            # (4) Append results
            # INDEX: 
            #       1 - X won
            #       2 - Tie
            #       3 - O won
            #
            if log in results.keys():
              results[log][i-1] += 1
            else:
              results[log] = [0,0,0]
              results[log][i-1] += 1


        except Exception as ins:
            print(f'Game {x}/{Ngames} failed with code: ', ins.args)


    # (5) Save
    with open('past.pkl','wb') as w: pickle.dump(past,w)
    if enhace: saver(results,f'_enhace_{perspective}')
    else: saver(results,'')    
    return

