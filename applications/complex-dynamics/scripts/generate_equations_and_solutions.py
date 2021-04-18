import sys
import pickle

import numpy as np
from sympy.solvers import nonlinsolve, solve, solveset

from sympy.geometry import Point
from sympy.core.symbol import symbols
from sympy import Symbol, Eq

def find_zeros(p: list):

  #*** this is the equation!
  #              0,5       1,6       2,7         3,8        4,9
  # d(x) = dt * (a_1 * x + b_1 * y + c_1 * x^2 + d_1 * x*y + e_1 * y^2) 
  # d(y) = dt * (a_2 * x + b_2 * y + c_2 * x^2 + d_2 * y*x + e_2 * y^2) 

  x, y = symbols('x y')
  Z = Point(x,y)
  r = solve([
             Z[0] * p[0] + Z[1] * p[1] + (Z[0]**2) * p[2] + (Z[0]**2) * p[4] + (Z[0]*Z[1]) * p[3],
             Z[0] * p[5] + Z[1] * p[6] + (Z[0]**2) * p[7] + (Z[1]**2) * p[9] + (Z[0]*Z[1]) * p[8],
                  ],Z,real=True)
  output = []
  for x in r:
    if x[0].is_real and x[1].is_real: output.append(x)
  if len(output)==1: output.append(output[0])
  return tuple(output)

def eval(p: list, Z: list):
  """
  auxiliary function for testing the roots that "find_zeros" finds!
  """
  return  (Z[0] * p[0] + Z[1] * p[1] + (Z[0]**2) * p[2] + (Z[0]**2) * p[4] + (Z[0]*Z[1]) * p[3],
             Z[0] * p[5] + Z[1] * p[6] + (Z[0]**2) * p[7] + (Z[1]**2) * p[9] + (Z[0]*Z[1]) * p[8])

def view(i):
  """
  auxiliary function for testing the roots that "find_zeros" finds!
  """
  try:
    with open('equations_and_solutions.pkl','rb') as w: data=pickle.load(w)
  except:
    with open('database/equations_and_solutions.pkl','rb') as w: data=pickle.load(w)
  k = list(data.keys())
  for j in range(len(data[k[i]])):
    print(f'root N {j} is {data[k[i]][j]} and evals to {eval(k[i],data[k[i]][j])}')


def generate_parameters(size: int = 10):
  return tuple(np.random.rand()*np.random.choice([-1,1],size))


def generator(TIMES):
  from datetime import datetime
  from dateutil.relativedelta import relativedelta
  data = {}
  ta = datetime.now()
  for _ in range(TIMES):
    if _%20==0: 
      tb = datetime.now()
      dt = relativedelta(tb,ta)
      with open('time-logs.txt','a') as f: 
        f.write(f'Lap {_} of {TIMES} took {dt.seconds+dt.microseconds/1e6} seconds\n')
      ta = datetime.now()
    p = generate_parameters()
    z = find_zeros(p)
    data[tuple(p)] = z
  try:
    with open('database/equations_and_solutions.pkl','rb') as f:
      temp = pickle.load(f)
    with open('database/equations_and_solutions.pkl','wb') as f:
      pickle.dump({**data,**temp},f)
  except:
    with open('database/equations_and_solutions.pkl','wb') as f:
      pickle.dump(data,f)
 
if __name__=='__main__':

  TIMES = 2000

  # generator: ON
  generator(TIMES)

  # viewer: OFF
  if False:
    for x in np.random.choice(range(TIMES),5,replace=False):
      view(x)  


