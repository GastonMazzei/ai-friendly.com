#!/usr/bin/env python

import os

import pandas as pd

from itertools import chain
from numpy import sin,sinh,cos,sqrt,abs
from numpy.random import randint
from random import choice

#------------------------------FORMAT SPECS---------------------------.
#                                                                     |
#---------------------------"LHCO" COLUMN FORMAT----------------------.
#  1    2    3       4    5      6     7      8      9        10   11 |
# (#) (typ) (eta) (phi) (pt) (jmass) (ntrk) (btag) (had/em) [dummy]   |
#                                                              [dummy]|
#----------------------------OUR ".CSV" FILES-------------------------.                  
#  .same but (tag_n) for n=12                                         |
#                                                                     |
#  .collisions involving m<n particles have (n-m)*11 blank cols       |
#                                                                     |
#  .Finally there is one last column that marks if Higgs              |
#---------------------------------------------------------------------.


#-------------------FIRST STEP--------------.
# 1) remove columns after "ntrk" (included) |
#    i.e. they are not kinematic info       |
#-------------------------------------------/

# FIRST STEP - Main
def remove_post_ntrk(df):
  """
  for df's under the 132+1-col
  spec defined in the current
  work!
  i.e. NOT for 12-col LHCO's
  """
  if len(df.columns)==133: pass
  else: print('wrong format')
  # columns are string-numbered (i.e. '1','2',..)
  cases = list(range(2,6))
  wanted = [[str(x+11*(i)) for x in cases] for i in range(12)]  
  wanted = list(chain.from_iterable(wanted))
  wanted += ['132']
  df.columns = [str(x) for x in df.columns]
  return df[wanted]

#--------------------SECOND STEP---------------------.
#                                                    |
#   http://madgraph.phys.ucl.ac.be/Manual/lhco.html  |
#                                                    |
#   "eta": pseudorapidity                            |
#                                                    |
#   "phi": azimuth                                   |
#                                                    |
#   "pt":  transverse momentum                       |
#                                                    |
# "jmass": invariant mass                            |
#                                                    |
#                                                    |
#  Pz = Pt * sinh(pseudorapidity)                    |
#  Px = Pt * cos(azimuth)                            |
#  Py = Pt * sin(azimuth)                            |
#                                                    |
#  and...                                            |
#  P0 / {M^2 = abs|P0^2 - Px^2 -Py^2 -Pz^2|}         |
#  then P0 = sqrt(M^2 - ||Pvect||^2)                 |
#----------------------------------------------------/

# SECOND STEP - Uils_Application
def application(_,dat):
  # for columns:
  # reindex to 0-3
  # 0:PseudoRap   1:Azimuth
  # 2:Pt          3:Jmass
  dat.columns = [aux for aux in range(len(dat.columns))]
  if _==0:
    # Px = Pt * cos(azimuth)
    serie = dat[2] * dat[1].apply(cos)  
  elif _==1:
    # Py = Pt * sin(azimuth)
    serie = dat[2] * dat[1].apply(sin)
  elif _==2:
    # Pz = Pt * sinh(pseudorapidity)
    serie = dat[2] * dat[0].apply(sinh)
  elif _==3:  
    # P0 = sqrt(M^2 - ||Pvect||^2)
    X = dat[2] * dat[1].apply(cos)
    Y = dat[2] * dat[1].apply(sin)
    Z = dat[2] * dat[0].apply(sinh)
    serie = (dat[3]**2 - X**2-Y**2-Z**2).apply(abs).apply(sqrt)
  return serie

# SECOND STEP - Main
def convert_to_4vect(df):
  """
  for df's under spec 72+1 rows
  which is the file under the
  132+1-col spec format
  without the >ntrk columns
  """
  # rename columns for successive indexing
  df.columns = list(range(len(df.columns)))
  c = df.copy()
  for i in range(12):
    localcols = [x for x in range(4*i,4*i+4)]
    for j in range(4):
      c[j+4*i] = application(j,df[localcols])
  return c

def opener(nm):
  d = pd.read_csv('database/'+nm,header=None, low_memory=False)
  return d

def save_spherical(df,naim):
  df.to_csv('database/spherical-'+naim,header=None,index=False)  
  return

def save_cartesian(df,naim):
  df.to_csv('database/cartesian-'+naim,header=None,index=False)  
  return


def main(nm):
  # open file
  d = opener(nm)

  # First Step
  d = remove_post_ntrk(d)
  save_spherical(d,nm)
  #print(d.head())
  print('\n\nFIRST STEP: so far so good!')
  
  # Second Step 
  d = convert_to_4vect(d)
  save_cartesian(d,nm)
  print('\n\nSECOND STEP: so far so good!')

if __name__=='__main__':
  for NAME in os.listdir('database'):
    try:
      print('starting ',NAME)
      main(NAME)
      print('ended case: ',NAME)
    except: pass

