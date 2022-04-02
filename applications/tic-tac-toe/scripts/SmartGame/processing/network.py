#!/usr/bin/env python
# coding: utf-8

"""Neural Network fits the data and saves trained-models
"""

import pickle

import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from SmartGame.processing.symmetry_rotation import rotate_to_bottom_left_center_of_mass as rblcm

def load(name):
    return pd.read_csv(name,low_memory=False).dropna().sample(frac=1)


def preprocess(df):
    data = {}
    print(df.head())
    x, y = df.to_numpy()[:,:-1].reshape(-1,len(df.columns)-1), df.to_numpy()[:,-1].reshape(-1,1)

    #---------------------------------------------------------------------------------------
    # here we apply the rotation symmetry
    length = x.shape[1]//2
    side = int(np.sqrt(length))
    opt = [1,2][0]
    # this "opt" mechanism allows
    # alternative applications
    if opt==1:
      z1 = np.apply_along_axis(lambda x: np.asarray(
                                               rblcm( 
                                             np.matrix(x.reshape(side,side))
                                                      )[0]
                                                       ).reshape(length,), 1, x[:,:length])
      z2 = np.apply_along_axis(lambda x: np.asarray(
                                               rblcm( 
                                             np.matrix(x.reshape(side,side))
                                                      )[0]
                                                       ).reshape(length,), 1, x[:,length:])
      x = np.concatenate([z1,z2],1)      
    #---------------------------------------------------------------------------------------

    if True:
      for s in [StandardScaler]:
        x=s().fit_transform(x)
    L = df.shape[0]
    divider = {'train':slice(0,int(0.7*L)),
               'val':slice(int(0.7*L),int((0.7+0.15)*L)),
               'test':slice(-int(0.15*L),None),}
    for k,i in divider.items():
        data[k] = (x[i],y[i])
        print(f'for key {k} {np.count_nonzero(data[k][1])/len(data[k][1])*100}% are non-zero')
        print(f'{data[k][0].shape}')
    return data
        
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
                kwargs.get('neurons',20),
                input_shape=(data['test'][0].shape[1],),
                activation=act,),
            Dense(
                kwargs.get('neurons',20),
                activation=act,),
            Dense(
                kwargs.get('neurons',20),
                activation=act,),
            Dense(
                1,
                activation='sigmoid'),
                    ]
    model = Sequential(architecture)
    model.compile(
                optimizer=SGD(learning_rate=kwargs.get('learning_rate',.01)),
                loss='binary_crossentropy',
                metrics='accuracy',)
    #
    # 2) Fit
    callback = EarlyStopping(monitor='loss', patience=5)
    results = model.fit(
            *data['train'],
            batch_size=kwargs.get('batch_size',32),
            epochs=kwargs.get('epochs',50),
            verbose=0,callbacks=[callback],
            validation_data=data['val'],)
    try: os.mkdir('./../data/models')
    except: pass 
    saving_name = kwargs.get('saving_name','') 
    model.save(f'./../data/models/model-{kwargs.get("model","o")}{saving_name}')

    #
    # 3) Return results
    results = results.history 
    results['ytrue_val'] = data['val'][1]
    results['ytrue_test'] = data['test'][1]
    results['ypred_val'] = model.predict(data['val'][0])
    results['ypred_test'] = model.predict(data['test'][0])
    results['specs'] = kwargs
 
    #
    # 4) Maybe, plot
    if kwargs.get('plot',False):
        regression = False
        case = 'val'
        f, ax = plt.subplots(1,3)
        if not regression:
          fpr, tpr, treshold = roc_curve(
                results['ytrue_'+case], results['ypred_'+case]
                    )
          ax[0].plot(tuple(fpr), tuple(tpr),label='AI-Friendly')
          fpr, tpr, treshold = roc_curve(
                results['ytrue_'+case], np.random.permutation(results['ytrue_'+case])
                    )
          ax[0].plot(tuple(fpr), tuple(tpr),label='random')
          ax[0].legend()

          weights = {0:[],1:[]}
          for i,x in enumerate(results['ypred_'+case]):
            weights[data[case][1][i][0]] += [x[0]]

       
          ax[1].hist(weights[0],label='0',alpha=0.5)
          ax[1].hist(weights[1],label='1',alpha=0.5)
          ax[1].set_xlim(0,1)
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

if __name__=='__main__':
    # RUN FROM CONSOLE WITH ARGUMENTS AND FIT A NEURAL NETWORK!
    # example:   
    #           python3 network.py 20 50 
    #
    #  and that fits a neural network of 20 neurons/layer and
    #  50 epochs over the data. At the end, results are plotted!
    #
    perspective = 'x'
    filename = f'./../data/processed-{perspective}.csv'
    print(f'Using the file {filename}; defaults to not-an-enhaced-network')
    create_and_predict(preprocess(load(filename),),
            neurons=int(sys.argv[1]), epochs=int(sys.argv[2]),plot=True, model=perspective)
