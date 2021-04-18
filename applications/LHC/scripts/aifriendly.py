#!/usr/bin/env python
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras import Sequential, backend
from keras.losses import binary_crossentropy
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from sklearn.metrics import roc_curve, roc_auc_score as ras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from random import randrange
from math import ceil

def ren(nu):
  # seed renormalizer
  # for Pandas seed<2**32 
  # constraint
  while nu>2**32:
    nu /= 2
  return int(nu) 

def get_seed():
  # setting up a permaSeed
  try:
    with open('seed','r') as f:
      for x in f:
        seed = int(x)
        if seed:
          break
        else: 
          raise
  except Exception as ins:
    print(ins.args)
    seed = ren(randrange(sys.maxsize))
    print(seed)
    with open('seed','w') as f:
      f.write(str(seed))
  return seed
 

def alternator(df):
  df1 = df[df.iloc[:,-1]==1]
  df2 = df[df.iloc[:,-1]==0]
  df1.index = [(2*x) for x in range(6650)]
  df2.index = [(2*x+1) for x in range(6650)]
  d = pd.concat([df1, df2]).sort_index()

  return (d)


def loader(nm):
  q = pd.read_csv(nm,low_memory=False,
         header=None)  
  if ((len(q) == 13300) and (
        len(q[q.iloc[:,-1]==1])==len(q)//2)):
    d = alternator(q)
    print('successfully alternated database!')
    print(d.head())
  else:
    d = q.sample(frac=1,
          random_state=get_seed())

  return d



def buildbasics(ps,name):
  data = {}
  names = ['train','val','test']
  for i,x in enumerate(names): 
    temp = loader(f'database/{name}{x}-{ps[i]}.csv')
    data[x] = (temp.iloc[:,:-1].to_numpy(dtype=float),temp.iloc[:,-1].to_numpy(dtype=int)) 

  # SCALING BUG!
  #print(f'\nBEFORE the fix we had {[data[x][0].shape for x in data.keys()]} \n')
  #print(f'\na little view is.. {[data[x][0][:12] for x in data.keys()]}\n')
  s = StandardScaler()
  s.fit(np.concatenate([data[x][0] for x in data.keys()]))
  for x in data.keys():
    data[x] = (s.transform(data[x][0]),data[x][1])
  #print(f'\nAFTER the fix we had {[data[x][0].shape for x in data.keys()]} \n')
  #print(f'\na little view is.. {[data[x][0][:12] for x in data.keys()]}\n')
  return data

def probachecker(v):
  for x in v: 	
    if ((str(x)!='2') and (str(x)!='50')):
      raise KeyError('ERROR: probas should be "2" or "50"')

  return v+[v[-1]] 

def main(**kwargs):
  """
  Bring the data
  Build the network
  Train the network
  Evaluate the network
  """
  
  #1) Params
  batch = kwargs.get('batch',32)
  epochs = kwargs.get('epochs',100)
  neurons = kwargs.get('neurons',32)
  ptrain = kwargs.get('ptrain',0.5)
  pval = kwargs.get('pval',0.5)
  verbose = kwargs.get('verbose',1)
  name = kwargs.get('name','')
  if name==0: name = ''
  if kwargs.get('deep',False): deep = 'deep-'
  else: deep=''
  #2) Data
  # should be a class and they should be concatenated
  # e.g. probaChecker([...,...,]).buildBasics()
  data = buildbasics(probachecker([ptrain,pval]), name)
  if name=='': name = 'full'

  #3) Build the network
  architecture = [Dense(neurons,activation='relu',
                  input_shape=(len(data['train'][0][0]),)),
                  Dense(neurons,activation='relu'),]
  if deep: 
    architecture += [Dense(neurons,activation='relu')] * 2
    print('2 extra layers')
  architecture += [Dense(1,activation='sigmoid'),]
  model = Sequential(architecture)
  model.compile(optimizer=SGD(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy'])
  if verbose: print('Network successfully built!')  

  #4) Train the network
  history = model.fit(data['train'][0], data['train'][1],
        batch_size = batch, epochs = epochs, verbose=verbose,
        validation_data=(data['val'][0], data['val'][1]))

  #5) Evaluate the network
  #history = pd.DataFrame(history.history)
  history = history.history
  history['ypred'] = model.predict(data['test'][0])
  history['ytrue'] = data['test'][1]
  history['info'] = (neurons,batch,epochs)
  #6) Return all results

  if deep:
    return history, f'hiddens-4_type-{name}',(data['train'],batch)
  else:
    return history, f'hiddens-2_type-{name}',(data['train'],batch)
  
def len_error(A,B):
  mssg = f"""
  WARNING: prediction
  and 'correct values' 
  are of different length!
  i.e. {len(B)} and {len(A)} 
  respect. 
       """
  raise(mssg)
  return

def pickle_bridge(A,B,C):
  info = A['info']
  first = f'{B}-E-{info[2]}-B-{info[1]}-N-{info[0]}'
  if False:
    # We are not saving each model's result
    with open(f"{first}.pickle", "wb") as f:
      pickle.dump(A, f)  

  return A,B,C

def sig_calc(a,ep):
  print(f'\n\nDEBUG-CHECKER: currently {ep} epochs?'\
         ' if so, ignore\n\n')
  batch, X, Y = a[1], a[0][0], a[0][1]
  training_data_len = 13300
  indexes = [i*batch for i in range(
            ceil(training_data_len/batch))]
  indexes_len = len(indexes)
  fractions = []
  k = 0
  for i in range(ep):
    if k==indexes_len-1:
      smth1 = indexes[k]
      smth2 = -1  
      fractions += [sum([j for j in Y[smth1:smth2] if j==1])/
                 len(Y[smth1:smth2])] 
      k = 0
    else:
      smth1 = indexes[k]
      smth2 = indexes[k+1]  
      fractions += [sum([j for j in Y[smth1:smth2] if j==1])/
                 len(Y[smth1:smth2])] 
      k += 1
  return fractions


def simple_plotter(h, plot_name, TR):
  # Build a plot
  f,ax = plt.subplots(1,3,figsize=(10,5))
  fs = 12

  #KILLFLAG # Left Plot: Training Performance
  #KILLFLAG #cases = ['accuracy', 'loss']
  #KILLFLAG #prefixes = ['', 'val_']
  ytrue, ypred, info = h['ytrue'], h['ypred'], h['info']
  del h['ytrue']
  del h['ypred']
  del h['info']
  h = pd.DataFrame(h)  
  h['ref 50%']=0.5
  # removed dataset insight!
  #newname = 'training signal %\nper epoch'
  #h[newname] = sig_calc(TR,len(h))

  #KILLFLAG for i_ in cases:
  #KILLFLAG  for j_ in prefixes:
  #KILLFLAG    sns.lineplot(y=j_+i_,ax=ax[0],data=h)#label=f'{j_}{i_}'
  sns.lineplot(data=h, ax=ax[0])
  ax[0].set_title(f'Loss & Accuracy vs Epochs',fontsize=fs)
  ax[0].set_xlabel('epochs',fontsize=fs)
  ax[0].set_ylim(0,1)
  
  # Middle Plot: Testing ROC
  fpr, tpr, treshold = roc_curve(ytrue,ypred)
  sns.lineplot(x=fpr,y=tpr,ax=ax[1],label=f'model - AUC={round(ras(ytrue,ypred),2)}')
  sns.lineplot(x=fpr,y=fpr,ax=ax[1],label='null')
  ax[1].set_ylim(0,1)
  ax[1].set_xlim(0,1)
  ax[1].set_title(f'ROC curve',fontsize=fs)

  # Right Plot: Fede Lamagna Weights
  if len(ytrue)==len(ypred): L = len(ypred)
  else: len_error(ytrue,ypred)
  temp = pd.concat([pd.DataFrame({
      'prediction': [ypred[i] for i in range(L) if ytrue[i]==1],
       'type': 'positive' ,}),
         pd.DataFrame({
       'prediction': [ypred[i] for i in range(L) if ytrue[i]==0],
        'type': 'negative',})]) 
  for x in ['positive','negative']:
    sns.distplot(temp[temp['type']==x]['prediction'],
                 bins=[y/10 for y in range(11)],
                 ax=ax[2], kde=False,norm_hist=True,
                 label=x,)
  ax[2].set_xlim(0,1)
  ax[2].set_xlabel('Output Weights',fontsize=fs)
  #ax[2].set_ylabel('Density',fontsize=fs)
  ax[2].set_title('output weights per category',fontsize=fs)
  ax[2].axvline(x=0.5,c='r',lw=1,ls='-')

  # AddSubTitle & Save!
  #f.suptitle(f'PARAMETERS: Epochs-{info[2]} Batch-{info[1]}'\
  #         f' Neurons-{info[0]} Signals-{plot_name}',fontsize=fs+2)
  f.suptitle(plot_name, fontsize=fs+2)
  #figname = f'{plot_name}-E-{info[2]}-B-{info[1]}-N-{info[0]}.png'
  #plt.tight_layout()
  figname = f'results/{plot_name}.png'
  f.savefig(figname)
  
  # Clear Backend
  del ax,f
  plt.figure().clear()
  plt.close("all")
   
  return
  
def backend_wrapper(process):
  debug = True
  if debug: process()
  else:
    backend.clear_session()
    try:
      process()
      print('SUCCESS!')
    except Exception as ins:
      print('ERROR: unexpected. \n',ins.args)
      backend.clear_session()
  return
  


if __name__=='__main__':
  # RUN: check for command-line-input
  try:
    print(sys.argv)
    params = []
    for x in sys.argv[1:]:
      if x not in ['spherical','cartesian','clean']:
        params += [int(x)]
      else:
        params += [x+'-']
    [PTR, PVA,NEU, EPO, BA, ne, dp] = params
    
    print('succesfully accepted the parameters!')
    print(params)
    backend_wrapper(
                lambda: simple_plotter(*pickle_bridge(
                        *main(
                               ptrain= PTR, pval= PVA,
                               neurons=NEU, epochs=EPO, 
                               batch=BA,  verbose=1, 
                               name=ne, deep=dp 
                                                       ))))
  except Exception as ins:
    print(ins.args)
    print('exiting')
