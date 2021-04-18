#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[19]:


def load(name=False):
    if name: return pd.read_csv(name)
    df = pd.read_csv('database/database.csv')
    return df


def preprocess(df, scaler=1, optsingle=False):
    #
    # part 1) load and scale
    #
    data = {}
    if scaler: s = StandardScaler()
    else: s = MinMaxScaler()
    df = df.sample(frac=1)
    x, y = df.to_numpy()[:,:-1], df.to_numpy()[:,-1]
    x = s.fit_transform(x)
    #
    # part 2) split into three
    #
    #
    if optsingle: return {optsingle: (x,y.astype(int))}
    L = df.shape[0]
    divider = {'train':slice(0,int(0.7*L)),
               'val':slice(int(0.7*L),int((0.7+0.15)*L)),
               'test':slice(-int(0.15*L),None),}
    for k,i in divider.items():
        data[k] = (x[i],np.round(y[i]).astype(int))
    #
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
    architecture = [
            Dense(
                kwargs.get('neurons',32),
                input_shape=(3,),
                activation='relu',),
            Dense(
                kwargs.get('neurons',32),
                activation='relu',),
            Dense(
                1,
                activation='sigmoid'),
                    ]
    model = Sequential(architecture)
    model.compile(
                optimizer=SGD(learning_rate=kwargs.get('learning_rate',.003)),
                loss='binary_crossentropy',
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
    results['specs'] = kwargs
    #
    if kwargs.get('plot',False):
        case = 'test'
        f, ax = plt.subplots(1,3, figsize=(20,7))
        fpr, tpr, treshold = roc_curve(
                results['ytrue_'+case], results['ypred_'+case]
                    )
        ax[0].plot(fpr, tpr)
        ax[0].set_title('ROC curve')
        ax[1].hist(results['ypred_'+case])
        ax[1].set_xlim(0,1)
        ax[1].set_title('Output probabilities histogram')
        ax[2].plot(results['accuracy'],c='b',label='train')
        ax[2].plot(results['val_accuracy'],c='g')
        ax[2].plot(results['loss'],c='b')
        ax[2].plot(results['val_loss'],c='g',label='validation')
        ax[2].legend()
        ax[2].set_title('Training metrics')
        ax[2].set_ylim(0,1)
        plt.savefig('network-results.png')
    return results


if __name__=='__main__':
    import sys
    create_and_predict(preprocess(load(),), 
            neurons=int(sys.argv[1]), epochs=int(sys.argv[2]),plot=True)
