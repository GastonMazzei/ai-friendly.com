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
from sklearn.metrics import roc_curve, roc_auc_score as ras
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, PowerTransformer, Binarizer


def load(name, what_to_fit, absolute=True):
    out = pd.read_csv(name,low_memory=False).dropna().sample(frac=1)[['field','T',what_to_fit]]
    if absolute: out[what_to_fit]=out[what_to_fit].apply(abs)
    print(out.head())
    return out

def preprocess(df, threshold,):
    data = {}
    x, y = df.to_numpy()[:,:-1].reshape(-1,len(df.columns)-1), df.to_numpy()[:,-1].reshape(-1,1)
    def split_according_to_threshold(y,threshold):
      effective_threshold=threshold
      L=len(y)
      result = sum(Binarizer(threshold= effective_threshold).fit_transform(y.reshape(-1,1)))/L
      tol = 0.01
      if not (threshold - tol <  result < threshold + tol):
        protect = 0
        while (not (threshold - tol <  result < threshold + tol)) and protect<100:
          if threshold-tol<result:
            effective_threshold += 0.01
          else:
            effective_threshold -= 0.01
          result = sum(Binarizer(threshold= effective_threshold).fit_transform(y.reshape(-1,1)))/L
          protect += 1
      return Binarizer(threshold= effective_threshold).fit_transform(y.reshape(-1,1))

    if True:
      for s in [StandardScaler]:
        x=s().fit_transform(x)
      y = split_according_to_threshold(y,threshold)    
    L = df.shape[0]
    divider = {'train':slice(0,int(0.7*L)),
               'val':slice(int(0.7*L),int((0.7+0.15)*L)),
               'test':slice(-int(0.15*L),None),}
    for k,i in divider.items():
        data[k] = (x[i],y[i])
        print(f'for key {k} the fractions of positives is{np.count_nonzero(data[k][1])/len(data[k][1])*100}%')
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
                kwargs.get('neurons',32),
                input_shape=(2,),
                activation=act,),
            Dense(
                kwargs.get('neurons',32),
                activation=act,),
            #Dense(
            #    kwargs.get('neurons',32),
            #    activation=act,),
            Dense(
                1,
                activation='sigmoid'),
                    ]
    model = Sequential(architecture)
    model.compile(
                optimizer=SGD(learning_rate=kwargs.get('learning_rate',.01)),
                loss='mean_squared_error',
                metrics='accuracy',)
    #
    # 2) Fit
    results = model.fit(
            *data['train'],
            batch_size=kwargs.get('batch_size',32),
            epochs=kwargs.get('epochs',50),
            verbose=1,callbacks=[EarlyStopping()],
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
        from sklearn.linear_model import LogisticRegression as lr
        f, ax = plt.subplots(1,3, figsize=(20,7))
        fpr, tpr, treshold = roc_curve(
              results['ytrue_'+case], results['ypred_'+case]
                  )
        ax[0].plot(tuple(fpr), tuple(tpr), label = 'NN AUC '+ str(
                                            round(
                                         ras(results['ytrue_'+case], results['ypred_'+case]),2)))
        if False:
          # Logistic Regression
          newytrue, newypred = data[case][1], lr(max_iter=5000).fit(*data['train']).predict_proba(data[case][0])[:,1]
          fpr2, tpr2, treshold = roc_curve(
              newytrue, newypred
                  )
          ax[0].plot(tuple(fpr2), tuple(tpr2), label = 'Logistic AUC '+ str(
                                            round(
                                         ras(newytrue, newypred), 2)))
        ax[0].set_title('ROC curve')
        ax[0].legend()
        weights = {0:[],1:[]}
        for i,x in enumerate(results['ypred_'+case]):
          weights[data[case][1][i][0]] += [x[0]]
     
        ax[1].hist(weights[0],label='0',alpha=0.5)
        ax[1].hist(weights[1],label='1',alpha=0.5)
        ax[1].set_xlim(0,1)
        ax[1].set_title('Output Weights per category')
        ax[1].legend()

        ax[2].plot(results['accuracy'],c='b',label='train')
        ax[2].plot(results['val_accuracy'],c='g')
        ax[2].plot(results['loss'],c='b')
        ax[2].plot(results['val_loss'],c='g',label='validation')
        ax[2].legend()
        ax[2].set_title('Training & Validation accuracy & loss')
        ax[2].set_ylim(0,1)
        f.suptitle(f'case: {kwargs["what_to_fit"]}')
        plt.savefig(f'gallery/network-result-{kwargs["what_to_fit"]}')
    return results

if __name__=='__main__':
    # --argv 1-- DEFAULT NEURONS SHOULD BE 2
    # --argv 2-- DEFAULT EPOCHS COULD BE 50 
    # --argv 3-- WHAT TO FIT CAN BE 1,2,3,4: 'M','X','C','E' respectively...
    # --argv 4-- THRESHOLD IS WHAT FRACTION OF POSITIVES WE TRAIN WITH
    what_to_fit_dictionary = {'1':'M','2':'X','3':'C','4':'E'}
    import sys                          
    create_and_predict(preprocess(load('dataset/ising.csv',what_to_fit_dictionary[sys.argv[3]]), float(sys.argv[4])),
            neurons=int(sys.argv[1]), epochs=int(sys.argv[2]), plot=True, what_to_fit=what_to_fit_dictionary[sys.argv[3]])


