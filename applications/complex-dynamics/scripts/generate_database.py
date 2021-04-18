#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle

def matrix_calculator(p, x):
  """ 
  only for order 2!
  """
  #              0,5       1,6       2,7         3,8        4,9
  # d(x) = dt * (a_1 * x + b_1 * y + c_1 * x^2 + d_1 * x*y + e_1 * y^2) 
  # d(y) = dt * (a_2 * x + b_2 * y + c_2 * x^2 + d_2 * y*x + e_2 * y^2) 
  return    np.asarray([
              [  p[0]+2*p[2]*x[0]+p[3]*x[1],
                 p[1]+2*p[4]*x[1]+p[3]*x[0],],

              [  p[5]+2*p[7]*x[0]+p[8]*x[1],
                 p[6]+2*p[9]*x[1]+p[8]*x[0],],
                             ], dtype=np.float32)
                

def convergence_veredict(matrix):
  eigs = np.linalg.eigvals(matrix)
  realparts = []
  for x in eigs:
    if np.isreal(x): realparts.append(x)
    else: realparts.append(x.real)
  if max(realparts)>0: return 1
  else: return 0

def rule(p,x):
  return convergence_veredict(matrix_calculator(p,x))

if __name__=='__main__':

  # (1) initialize
  data = {}
  names = [
           #'x0','y0',
              'a_1','b_1','c_1','d_1','e_1',
              'a_2','b_2','c_2','d_2','e_2',
                                       'divergence',]
  for x in names: data[x] = []

  # (2) load equations and solutions
  with open('database/equations_and_solutions.pkl','rb') as f:
    equations_and_solutions = pickle.load(f)

  # (3) append the convergence veredict for every case
  for k in equations_and_solutions.keys():
    counter = 0
    for j in equations_and_solutions[k]:        
      counter += rule(k, j)
    if counter>0: counter=1
    param = k + (float(counter),)
    for i,x in enumerate(names): data[x] += [param[i]]

  # (4) balance dataset
  df = pd.DataFrame(data)
  q = df['divergence'].value_counts()
  m = np.argmin([q[i] for i in range(2)])
  df = pd.concat([df[df['divergence']==m], df[~(df['divergence']==m)][:int(q[m]*1.25)]],0).sample(frac=1)

  # (5) save
  df.to_csv('database/database.csv', index=False)
