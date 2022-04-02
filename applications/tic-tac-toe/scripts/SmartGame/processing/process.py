#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""process:
   retrieves the game-results ('.pkl')
   and generates two '.csv': one per
   each possible perspective
"""

import os
import pickle
import random
import re
import sys
import time

import pandas as pd
import numpy as np
  
from math import sqrt

def processer(dat, perspective: str='o'):
  l = []
  k = list(dat.keys())
  L = int(sqrt(len(k[0][0]))) 
  print('L is ',L,'!')
  if perspective=='o': 
    f = {0:0,1:1,2:1}
    def r(x):
      if len(x)%2: return slice(0,None)
      else: return slice(0,-1)
  else: 
    f = {0:1,1:1,2:0}
    def r(x):
      if len(x)%2: return slice(0,-1)
      else: return slice(0,None)

  for x in dat.keys():
    if perspective=='o':
      v = x[r(x)]
    else:
      v = (tuple([-1]*(L**2)),*x[r(x)])
    for j_ in range(3):
      for q in range(dat[x][j_]):
        for i in range(len(v)//2):
          l.append( (*v[2*i],*v[2*i+1],f[j_]) )
  return pd.DataFrame(np.asarray(l))        
      
def processer_3x3_backup(dat, perspective: str='o'):
  l = []
  if perspective=='o': 
    f = {0:0,1:1,2:1}
    def r(x):
      if len(x)%2: return slice(0,None)
      else: return slice(0,-1)
  else: 
    f = {0:1,1:1,2:0}
    def r(x):
      if len(x)%2: return slice(0,-1)
      else: return slice(0,None)

  for x in dat.keys():
    if perspective=='o':
      v = x[r(x)]
    else:
      v = (tuple([-1]*9),*x[r(x)])
    for j_ in range(3):
      for q in range(dat[x][j_]):
        for i in range(len(v)//2):
          l.append( (*v[2*i],*v[2*i+1],f[j_]) )
  return pd.DataFrame(np.asarray(l))        


def init(opt=False):
  if opt:
    with open(f'./../data/results_enhace_{opt}.pkl','rb') as w: 
      data = pickle.load(w)
    return data
  with open('./../data/results.pkl','rb') as w: 
    data = pickle.load(w)
  return data

