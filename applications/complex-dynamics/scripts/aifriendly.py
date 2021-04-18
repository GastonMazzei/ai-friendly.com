#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, PowerTransformer, Binarizer

def load(name):
    return pd.read_csv(name,low_memory=False).dropna().sample(frac=1)

def preprocess(df):

    data = {}
    L = df.shape[0]
    x, y = (df.to_numpy()[:,:-1].reshape(-1,len(df.columns)-1),
                                      df.to_numpy()[:,-1].reshape(-1,1))

    if True: #False
      for s in [MinMaxScaler]:
        x=s().fit_transform(x)

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
 
    # 1) Initialize
    act = 'relu'
    architecture = [
            Dense(
                kwargs.get('neurons',32),
                input_shape=(data['train'][0].shape[1],),
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

    # 2) Fit
    results = model.fit(
            *data['train'],
            batch_size=kwargs.get('batch_size',10),
            epochs=kwargs.get('epochs',50),
            verbose=1,callbacks=[EarlyStopping()],
            validation_data=data['val'],)

    # 3) return results
    results = results.history 
    results['ytrue_val'] = data['val'][1]
    results['ytrue_test'] = data['test'][1]
    results['ypred_val'] = model.predict(data['val'][0])
    results['ypred_test'] = model.predict(data['test'][0])
    results['specs'] = kwargs

    # 4) plot
    if kwargs.get('plot',False):
        from sklearn.metrics import roc_auc_score as ras
        regression = False
        case = 'test'
        f, ax = plt.subplots(1,3, figsize=(15,5))
        fpr, tpr, treshold = roc_curve(
              results['ytrue_'+case], results['ypred_'+case]
                  )
        ax[0].plot(fpr, tpr, label=f"NN AUC-{ras(results['ytrue_'+case],results['ypred_'+case])}")
        ax[0].plot([0,1],[0,1],label='random',c='y')
        ax[0].legend()
        weights = {0:[],1:[]}
        for i,x in enumerate(results['ypred_'+case]):
          weights[data[case][1][i][0]] += [x[0]]
      
        ax[1].hist(weights[0],label='0',alpha=0.5)
        ax[1].hist(weights[1],label='1',alpha=1)
        ax[1].set_xlim(0,1)
        ax[1].legend()

        ax[2].plot(results['accuracy'],c='b',label='train')
        ax[2].plot(results['val_accuracy'],c='g')
        ax[2].plot(results['loss'],c='b')
        ax[2].plot(results['val_loss'],c='g',label='validation')
        ax[2].legend()
        ax[2].set_ylim(0,1)
        plt.show()
    return results

if __name__=='__main__':
    import sys
    create_and_predict(preprocess(load('database/database.csv'),),
            neurons=int(sys.argv[1]), epochs=int(sys.argv[2]),plot=True)
