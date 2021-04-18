import numpy as np
import pandas as pd
# Imports de estadistica
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
# Imports para guardar el modelo como un file
from sklearn.externals import joblib
# Imports para las redes de Keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import backend

class AI_model:
    def __init__(self, dataset, model=False):
        dimensions = dataset.shape[1]-1
        X = dataset[:,0:dimensions]
        Y = dataset[:,dimensions]
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        a,b,c,d   = train_test_split(X_scale, Y, test_size=0.3)
        (X_train, X_val_and_test,
                 Y_train, Y_val_and_test) = (a,b,c,d) 
        (e,f,g,h) = train_test_split(X_val_and_test,
                                  Y_val_and_test,test_size=0.5)
        (X_val, X_test,Y_val,Y_test) = (e,f,g,h)
        # CUSTOM NETWORK ARCHITECTURE
        #activation = model[-1]
        #optimizer = model[-2]
        optimizer = 'SGD'
        #model_layers = [int(x) for x in model[:-2]]
        #model_definer = []
        #for i in range(len(model_layers)):
        #   if i==0:   
        #       model_definer += [Dense(model_layers[i], activation=activation, input_shape=(dimensions,))]
         #   else: 
        #        model_definer += [Dense(model_layers[i], activation=activation)]
        #model_definer += Dense(1, activation='sigmoid')
        
        #self.model = Sequential(model_definer)  
        
        self.model = Sequential([
            Dense(32, activation='relu', input_shape=(dimensions,)),
            Dense(32, activation='relu'),
            #Dense(32, activation='relu'),
            #Dense(16, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])

        self.model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        self.dimensions = dimensions
        self.scaler = min_max_scaler
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.history = None
        self.confidence = None
        
    def trainModel(self):
        self.history = self.model.fit(self.X_train, self.Y_train,
                  batch_size=32, epochs=200, verbose=0,
                  validation_data=(self.X_val, self.Y_val))

        self.confidence = self.model.evaluate(self.X_test, self.Y_test, verbose=0)[1]    
        return self.model

