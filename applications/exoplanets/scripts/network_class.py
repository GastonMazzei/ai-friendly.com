#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import SGD

from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, PowerTransformer, Binarizer


# In[19]:


def load(name):
    return pd.read_csv(name,low_memory=False).dropna().sample(frac=1)


def preprocess(df):

    data = {}
    verbose = True

    df = df[df['PER'].between(0,20000)].sample(frac=1)
    print(df.head())
    x, y = df.to_numpy()[:,:-1].reshape(-1,1), df.to_numpy()[:,-1].reshape(-1,1)

    f,ax = plt.subplots(1,2)
    ax[0].hist(x)
    ax[0].set_title('x prev')
    ax[1].hist(y)
    ax[1].set_title('y prev')
    if verbose: plt.show()
    
    if False:
      for s in [MinMaxScaler]:#[PowerTransformer, StandardScaler]:
        x=s().fit_transform(x)
        y=s().fit_transform(y)

    f,ax = plt.subplots(1,2)
    ax[0].hist(x)
    ax[0].set_title('x post')
    ax[1].hist(y)
    ax[1].set_title('y post')
    if verbose: plt.show()
    else: plt.close()

    if False:
      answ = input('please insert the radius range: e.g. (x y)')
      rmin = float(answ.split(' ')[0])
      rmax = float(answ.split(' ')[1])
      y = np.asarray([[1] if rmin<x[0]<rmax else [0] for x in y])  
    elif True:
      y = Binarizer(threshold=np.mean(y)).fit_transform(y)    
      #y = y

    L = df.shape[0]
    divider = {'train':slice(0,int(0.7*L)),
               'val':slice(int(0.7*L),int((0.7+0.15)*L)),
               'test':slice(-int(0.15*L),None),}

    for k,i in divider.items():
        data[k] = (x[i],y[i])
        print(f'for key {k} {np.count_nonzero(data[k][1])/len(data[k][1])*100}% are non-zero')

    answ = input('if you are happy with the ratio, press "y"... else "n"')
    if answ=='y': return data
    else: return preprocess(df)
        
def manage_database(name):
  with open(name,'rb') as f:
    data = pickle.load(f)
  return
    
def create_and_predict(data,**kwargs):
    """
    kwargs: 
        neurons=32
        epochs=50
        learning_rate=0.01
        batch_size=32
        plot=False
    """
    #
    # 1) Initialize
    act = 'relu'
    architecture = [
            Dense(
                kwargs.get('neurons',32),
                input_shape=(1,),
                activation=act,),
            Dense(
                kwargs.get('neurons',32),
                activation=act,),
            Dense(
                1,
                activation='sigmoid'),
                    ]
    model = Sequential(architecture)
    model.compile(
                optimizer=SGD(learning_rate=kwargs.get('learning_rate',.001)),
                loss='mean_squared_error',
                metrics='accuracy',)
    #
    # 2) Fit
    results = model.fit(
            *data['train'],
            batch_size=kwargs.get('batch_size',50),
            epochs=kwargs.get('epochs',50),
            verbose=1,
            validation_data=data['val'],)
    #
    # 3) return results
    results = results.history 
    results['ytrue_val'] = data['val'][1]
    results['ytrue_test'] = data['test'][1]
    results['ypred_val'] = model.predict(data['val'][0])
    results['ypred_test'] = model.predict(data['test'][0])
    results['specs'] = kwargs
    #
    if kwargs.get('plot',False):
        regression = False
        case = 'val'
        f, ax = plt.subplots(1,3)
        if not regression:
          fpr, tpr, treshold = roc_curve(
                results['ytrue_'+case], results['ypred_'+case]
                    )
          ax[0].plot(fpr, tpr)
        
          weights = {0:[],1:[]}
          for i,x in enumerate(results['ypred_'+case]):
            weights[data[case][1][i][0]] += [x[0]]

          #sns.distplot(data=pd.DataFrame(weights),ax=ax[1],kde=False)
       
          ax[1].hist(weights[0],label='0',alpha=0.5)
          ax[1].hist(weights[1],label='1',alpha=1)
          ax[1].set_xlim(0,1)
          ax[1].legend()

        else:
          from scipy.optimize import curve_fit
          def f(x,a): return a*x**(3/2)
          #p, _ = curve_fit(f,data[case][0].flatten(),results['ypred_'+case].flatten())
          p, _ = curve_fit(f,data[case][0].flatten(),data[case][1].flatten())
          print(f'\n\n\n\n{p}\n\n\n\n')
          new_x = np.linspace(min(data[case][0]),max(data[case][0]),1000)
          ax[1].scatter(data[case][0],data[case][1],label='true values '+case,c='r')
          ax[1].scatter(data[case][0],results['ypred_'+case],label='predicted values '+case,c='b')
          ax[1].scatter(new_x,[f(q,p[0]) for q in new_x],label=r'$T \propto R^{\frac{3}{2}}$',c='g',alpha=0.5) 
          ax[1].set_xlabel('$\propto$ T')
          ax[1].set_ylabel('$\propto$ Radius')
          ax[1].legend()

        ax[2].plot(results['accuracy'],c='b',label='train')
        ax[2].plot(results['val_accuracy'],c='g')
        ax[2].plot(results['loss'],c='b')
        ax[2].plot(results['val_loss'],c='g',label='validation')
        ax[2].legend()
        ax[2].set_ylim(0,1)
        plt.show()
        if False:
            plt.plot(
                *roc_curve(
                    results['ytrue_test'], results['ypred_test']
                        )[:-1])
    return results


# In[ ]:

if __name__=='__main__':
    import sys
    create_and_predict(preprocess(load('petit-database.csv'),),
            neurons=int(sys.argv[1]), epochs=int(sys.argv[2]),plot=True)


