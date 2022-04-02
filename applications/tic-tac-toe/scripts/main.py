#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main:
   run once and it
   (1) generates games (default: random vs random, 10k)
   (2) process them into ".csv" files 
   (3) train a neural network with it and generate "model" files
"""

import os
import pickle
import random
import re
import sys
import time
import uuid

import pandas as pd
import numpy as np

from SmartGame.generating.classdef import TickTackToe
from SmartGame.generating.core import core
from SmartGame.processing.process import init, processer
from SmartGame.processing.network import load, preprocess, create_and_predict
from SmartGame.interactive.play import play
from SmartGame.interactive.play_x import play_x

def generator(ngames: list, grid: tuple, verbose: bool=False, enhace=False, perspective='',**kwargs):
    try:
        L, pL = grid
        if perspective:
          core(ngames,L,pL,verbose, enhace, perspective,**kwargs)
        else:
          core(ngames,L,pL,verbose, enhace)
    except Exception as ins:
        mssg = f'\n STATUS: an error ocurred'
        print(mssg, ins.args)
    return 


if __name__=='__main__':
    plot = [False,True][1] #<-- True or False 
    grid = (3,3) #<-- (L,pL) e.g. (3,3) 

    try:
        played=False
        if sys.argv[1]=='play': 
            played=True
            version = sys.argv[2]
            try: enhace=sys.argv[3]
            except: enhace=''
            try:
                raise Exception('not built yet')
                play(*[int(j_) for j_ in sys.argv[3:]])
            except:
                if version=='first':
                  if enhace: play(*grid,'enhace')
                  else: play(*grid)
                else: 
                  if enhace: play_x(*grid,'enhace')
                  else: play_x(*grid)
    except: 
        pass 

    if played:
        sys.exit(0)
    else:
        try:
          if sys.argv[1]=='enhace': enhace=True
        except: enhace=False

        # Generate games!
        verbose = False
        N = 8000
        perspective = ['o','x']

        # Process game-results!
        if enhace:
          noise = 0.5
          if False: 
            #network vs random...
            for _ in perspective:
              generator(N , grid, verbose, enhace, _,
                        random_O=noise, random_X=noise,)
              df = processer(init(_), _)
              df.to_csv(f'./../data/processed-{_}_enhace.csv',index=False)
          else:  
            #networks vs each other
            generator(N , grid, verbose, enhace, 'both',
                        random_O=noise, random_X=noise,)
            for _ in perspective:
              df = processer(init('both'), _)
              df.to_csv(f'./../data/processed-{_}_enhace.csv',index=False)
          # Fit a network!
          #
          for _ in perspective:    
            create_and_predict(preprocess(load(f'./../data/processed-{_}_enhace.csv'),),
                                          neurons=32, epochs=150, plot=plot, #batch_size=128,
                                          model=_, saving_name='_enhace')
          # Play!
          AGAINST = 'O'
          if AGAINST=='O':
            play(*grid, True)
          else:
            play_x(*grid, True)
        else:
          generator(N , grid, verbose, enhace)
          for _ in perspective:
              df = processer(init(), _)
              df.to_csv(f'./../data/processed-{_}.csv',index=False)
 
          # Fit a network!
          for _ in perspective:    
              create_and_predict(preprocess(load(f'./../data/processed-{_}.csv'),),
                  neurons=32, epochs=150, plot=plot, model=_)
          # Play!
          AGAINST = 'O'
          #.
          if AGAINST=='O':
            play(*grid)
          else:
            play_x(*grid)
