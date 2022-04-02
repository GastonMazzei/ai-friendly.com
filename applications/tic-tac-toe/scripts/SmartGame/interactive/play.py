import numpy as np

import random
import os
import numpy as np
import pandas as pd
import uuid
import time
import pandas as pd
import sys

from sklearn import preprocessing, metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense

from keras import backend
import os

from SmartGame.generating.classdef import TickTackToe
from SmartGame.interactive.brains import *

def play(L=3,pL=3, enhace=False):
  name = 'AI-Friendly'
  T = TickTackToe(L,pL)
  from keras.models import load_model
  from sklearn.preprocessing import StandardScaler
  TYPE = 'o'
  if enhace: 
    model = load_model(f'./../data/models/model-{TYPE}_enhace')
    scaler = StandardScaler().fit(pd.read_csv(f'../data/processed-{TYPE}_enhace.csv').to_numpy()[:,:-1])
    print('LOADED THE ENHACED VERSION!')
  else: 
    model = load_model(f'./../data/models/model-{TYPE}')
    scaler = StandardScaler().fit(pd.read_csv(f'../data/processed-{TYPE}.csv').to_numpy()[:,:-1])

  def INTRO():
      print(f'\n\n\n\nHola! Si estas aca es porque queres jugar contra {name}... Suerte!\n\n') 
      time.sleep(5)
      print('launching game in... \n3')
      time.sleep(1)
      print('  2')
      time.sleep(1)
      print('     1')
      time.sleep(1)
  #INTRO()
  timer = False
    
  try:
      i,j = request_and_return(T.board)
      T.board.itemset((i,j),1)
      if timer:
        time.sleep(2)
        print('mire como ha quedado el tablero:')
      tablero_printer(T.board)
      if timer:
        time.sleep(2)
        print(f'Ahora {name} pensara una respuesta...')
        print('.')
        time.sleep(0.5)
        print('..')
        time.sleep(0.5)
        print('...')
        time.sleep(0.5)
  except Exception as inst:
      print('Oops! inner problem encountered. Game Ended')
      print(inst.args)

  init = True
  message = ''
  while init:
      try:
          if timer:
            time.sleep(1)
          print('hi!')
          #T,message,init =respond(T,model,scaler,name,allowedMoves,L,pL)
          T, message, init = new_respond(T,model,scaler,name)
          if timer: 
            time.sleep(1)
      except Exception as ins:
          print(ins.args)
          print('--IT WAS A TIE--')
          tablero_printer(T.board)    
          break
      if init:
          print(message)
          i,j = request_and_return(T.board)
          T.board.itemset((i,j),1)
          if T.checkX():
              print("--HAS GANADO!--")
              tablero_printer(T.board)
              break 
      else:
          print(message)
          tablero_printer(T.board)
