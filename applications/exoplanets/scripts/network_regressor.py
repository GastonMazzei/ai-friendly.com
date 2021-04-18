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
from keras.callbacks import EarlyStopping

from sklearn.metrics import roc_curve, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, PowerTransformer, Binarizer



def load(name):
    return pd.read_csv(name,low_memory=False).dropna().sample(frac=1)

def preprocess(df):

    df=df.to_numpy()
    df = StandardScaler().fit_transform(df)
    data = {}
    for k,v in {'train':1700,'val':400,'test':len(df)-1700}.items():
      data[k] = (df[:v,:-1].reshape(-1,2),df[:v,-1].reshape(-1,1))
      df = df[v:,:]  
    return data

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
                input_shape=(2,),
                activation=act,),
            Dense(
                kwargs.get('neurons',32),
                activation=act,),
            Dense(
                1,
                activation='relu'),
                    ]
    model = Sequential(architecture)
    model.compile(
                optimizer=SGD(learning_rate=kwargs.get('learning_rate',.01)),
                loss='mean_squared_error',
                metrics='accuracy',)
    #
    # 2) Fit
    callback = EarlyStopping(monitor='loss', patience=5)
    results = model.fit(
            *data['train'],
            batch_size=kwargs.get('batch_size',32),
            epochs=kwargs.get('epochs',50),
            verbose=1,callbacks=[callback],
            validation_data=data['val'],)
    #
    # 3) return results
    results = results.history 
    results['ytrue_val'] = data['val'][1]
    results['ytrue_test'] = data['test'][1]
    results['ypred_val'] = model.predict(data['val'][0])
    results['ypred_test'] = model.predict(data['test'][0])
    print(f'THE RESULT FOR REGRESSION WAS: {mean_squared_error(results["ypred_test"],data["test"][1])} mse')
    predictions = pd.DataFrame({
                                'mass':data['test'][0][:,0].tolist()*2,
                                'radius':data['test'][0][:,1].tolist()*2,
                                'period':np.concatenate([data['test'][1], results['ypred_test']],0).flatten().tolist(),
                                'type':['true value']*len(data['test'][1])+ ['predicted value']*len(data['test'][1]),
                                 })
    predictions.to_csv('database/network-regression-predictions.csv',index=False)
    
    results['specs'] = kwargs
    #
    if kwargs.get('plot',False):
        regression = True
        case = 'test'
        f, ax = plt.subplots(1,3)
        if not regression:
          fpr, tpr, treshold = roc_curve(
                results['ytrue_'+case], results['ypred_'+case]
                    )
          ax[0].plot(fpr, tpr)
        
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
    create_and_predict(preprocess(load('database/filtered-database.csv'),),
            neurons=int(sys.argv[1]), epochs=int(sys.argv[2]),plot=False)


