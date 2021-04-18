# AI
import numpy as np
import pandas as pd
import sys
import json

from sklearn import preprocessing, metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D
from keras.optimizers import SGD, Adam

from keras import backend

# extras
import extras
# plots
import matplotlib.pyplot as plt

default_params = {
'layer1_basic':'',
'layer2_basic':'',
'activation_basic':'relu',
'epochs_basic':str(100),
'optimizer_advanced':'sgd',
'norm_advanced':None,
'quad_advanced':None,
'noise_advanced':None,
'timeseries_advanced':None,
'sine_advanced':None,
'timeseries_advanced':None,
'layer1-advanced':'',
'layer2-advanced':'',
'layer3-advanced':'',
'activation_advanced':'relu',
'epochs_advanced':str(100),
'batch_advanced':str(32),
'learning_rate_advanced':str(1),                 
                 }


def decider(d: dict):
    if (
       (d.get('learning_rate_advanced')!=default_params['learning_rate_advanced']) or
       (d.get('batch_advanced')!=default_params['batch_advanced']) or
       (d.get('epochs_advanced')!=default_params['epochs_advanced']) or
       (d.get('activation_advanced')!=default_params['activation_advanced']) or
       (d.get('layer3-advanced')!=default_params['layer3-advanced']) or
       (d.get('layer2-advanced')!=default_params['layer2-advanced']) or
       (d.get('layer1-advanced')!=default_params['layer1-advanced']) or
       (d.get('sine_advanced')!=default_params['sine_advanced']) or
       (d.get('noise_advanced')!=default_params['noise_advanced']) or
       (d.get('quad_advanced')!=default_params['quad_advanced']) or
       (d.get('norm_advanced')!=default_params['norm_advanced']) or
       (d.get('optimizer_advanced')!=default_params['optimizer_advanced']) or
       (d.get('timeseries_advanced')!=default_params['timeseries_advanced']) 
          ): return 2 
    elif (
       (d.get('epochs_basic')!=default_params['epochs_basic']) or
       (d.get('activation_basic')!=default_params['activation_basic']) or
       (d.get('layer2_basic')!=default_params['layer2_basic']) or
       (d.get('layer1_basic')!=default_params['layer1_basic'])
           ): return 1
    else: return 0


def flags_constructor(d: dict):
    flags = []
    check = ['sine_advanced','quad_advanced','norm_advanced']
    for x in check:
        if d.get(x)=='true': flags.append(x)
    return flags

def add_noise(x):
    u, s = np.mean(x,0), np.std(x,0)
    length,width = x.shape
    noise = np.zeros((length,width))
    for i in range(width):
        noise[:,i] += np.random.normal(u[i], s[i]/2, length)
    return x+noise

def scaler_constructor(flags: list):
    if ('sine_advanced' in flags) or ('quad_advanced' in flags):
        if ('sine_advanced' in flags) and ('quad_advanced' in flags):
            features = FeatureUnion([("sine", preprocessing.FunctionTransformer(np.sin)),
                          ("quadratic", preprocessing.FunctionTransformer(np.square))])
            print('a')
        elif ('sine_advanced' in flags):
            features = preprocessing.FunctionTransformer(np.sin)
            print('b')
        else:
            features = preprocessing.FunctionTransformer(np.square)
            print('c')
        if ('norm_advanced' in flags):
            scaler = Pipeline([ 
                             ('features', features),
                             ('norm', preprocessing.StandardScaler()),
                             ('final_operation', preprocessing.MinMaxScaler())
                                ])
            print('d')
        else:
            scaler = Pipeline([ 
                             ('features', features),
                             ('final_operation', preprocessing.MinMaxScaler())
                                ])
            print('e')
    elif ('norm_advanced' in flags):
        scaler = Pipeline([ 
                     ('norm', preprocessing.StandardScaler()),
                     ('final_operation', preprocessing.MinMaxScaler())
                           ])
        print('f')
    else:
        scaler = preprocessing.MinMaxScaler()        
        print('g')
    return scaler

def function_for_timeseries(x):
    return ((x-np.min(x))/(np.max(x)-np.min(x))).reshape(x.shape[0],1,x.shape[1])

class AImodel:
    def __init__(self, dataset, networkname):
        dimensions = dataset.shape[1]-1

        try:
            with open(networkname+'/NetworkParams.json', 'r') as f:
                self.params = json.load(f)
        except Exception as ins:
            print('\n\nFAILED TO LOAD CUSTOM PARAMETERS:\n\n\n',ins.args,'\n\n\n\n\n') 
            self.params = default_params

        # for debugging:
        print(f'\n\n\nparameters are: {self.params}\n\n\n')

        self.customization_level = decider(self.params)
        print('IT IS LEVEL ',self.customization_level)
     
        X = dataset[:,0:dimensions]
        Y = dataset[:,dimensions]

        if self.customization_level!=2:
            scaler = preprocessing.MinMaxScaler()
        else:
            if self.params.get('timeseries_advanced',None):               
                scaler = FunctionTransformer(func=function_for_timeseries)
            else:
                scaler = scaler_constructor(flags_constructor(self.params))
            
        print(f'before scaling data had MAX,MIN={np.max(X)}{np.min(X)} and shape {X.shape}')
        X_scale = scaler.fit_transform(X)
        print(f'after scaling data had MAX,MIN={np.max(X_scale)}{np.min(X_scale)} and shape {X_scale.shape}')

        if self.params.get('noise_advanced')=='true':
            X_scale = add_noise(X_scale)
        
        X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)


        if self.customization_level==0:
            self.model = Sequential([
                Dense(32, activation='relu', input_shape=(dimensions,)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid'),
            ])
        elif self.customization_level==1:
            self.model = Sequential([
                Dense(int((self.params['layer1_basic'] if self.params['layer1_basic']!='' else 16) if 'layer1_basic' in self.params else 8),
                      activation=self.params['activation_basic'],
                      input_shape=(dimensions,)),
                Dense(int((self.params['layer2_basic'] if self.params['layer2_basic']!='' else 16) if 'layer2_basic' in self.params else 8),
                      activation=self.params['activation_basic']),
                Dense(1, activation='sigmoid'),
            ])
        else:
            architecture = []
            if self.params.get('timeseries_advanced',None):
                architecture += [Conv1D(8,2,padding='same',activation=self.params['activation_advanced']),
                                 Conv1D(16,2,padding='same',activation=self.params['activation_advanced']),
                                 MaxPool1D(pool_size=2,strides=1,padding='same'),
                                 Flatten(),]

            architecture += [
                Dense(int((self.params['layer1_advanced'] if self.params['layer1_advanced']!='' else 16) if 'layer1_advanced' in self.params else 8),
                      activation=self.params['activation_advanced']),
                Dense(int((self.params['layer2_advanced'] if self.params['layer2_advanced']!='' else 16) if 'layer2_advanced' in self.params else 8),
                      activation=self.params['activation_advanced']),
                Dense(int((self.params['layer3_advanced'] if self.params['layer3_advanced']!='' else 16) if 'layer3_advanced' in self.params else 8),
                      activation=self.params['activation_advanced']),
            ]

            architecture.append(Dense(1, activation='sigmoid'))

            self.model = Sequential(architecture)
       
        if self.customization_level!=2:
            optimizer = 'sgd'
        else:
            print(float(self.params['learning_rate_advanced'])/100)
            if self.params['optimizer_advanced']=='sgd':
                optimizer = SGD(lr=float(self.params['learning_rate_advanced'])/100)
            elif self.params['optimizer_advanced']=='adam':
                optimizer = Adam(lr=float(self.params['learning_rate_advanced'])/100)

        self.model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.dimensions = dimensions
        self.scaler = scaler
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.history = None
        self.confidence = None
        
    def trainModel(self):

        if self.customization_level==0:
            batch_size, epochs = 32,100 
        elif self.customization_level==1:
            batch_size, epochs = 32, int(self.params['epochs_basic']) 
        else:
            batch_size, epochs = int(self.params['batch_advanced']), int(self.params['epochs_advanced'])
 
        self.history = self.model.fit(self.X_train, self.Y_train,
                  batch_size=batch_size, epochs=epochs, verbose=0,
                  validation_data=(self.X_val, self.Y_val))

        self.confidence = self.model.evaluate(self.X_test, self.Y_test, verbose=0)[1]    
        return self.model
    
    
def readDataFromFile(filename):
    df = pd.read_excel(filename, header=None)
    data = (df.to_numpy())
    
    try:
        [float(x) for x in data[0]]
        print('La primera fila parecen datos... no hay nombres de columna')
        columnNames = []
    except:
        print('Usando nombres para las columnas')
        columnNames = data[0]
        print(columnNames)
        df = pd.read_excel(filename, header=0)
        data = df.to_numpy()
        
    return data, columnNames 


def processInData(model, scaler, inDataset):
    inDataset_scale = scaler.fit_transform(inDataset)
    result = model.predict_classes(inDataset_scale)
    output = np.hstack((inDataset, result))
    
    return output
    
    
def process(dir):
    print(' ========== AI PROCESS ========== ')
    
    #model tasks
    LCFILE = extras.LearnCardFile(dir)
    dataset, columnNames = readDataFromFile( dir + r'/%s'%LCFILE )
    if columnNames != []: resultTag = columnNames[-1]
    else: resultTag = 'prediction'
    
    nNetwork = AImodel(dataset,dir)
    nNetwork.trainModel()
    nNetwork.model.save(dir + r'/model.h5')
    joblib.dump(nNetwork.scaler, dir + r'/scaler.dat') 
    
    result = {}
    result['confidence'] = nNetwork.confidence
    
    #process inCard
    ICFILE = extras.InCardsFiles(dir)[0]
    testdata, columnNames = readDataFromFile( dir + r'/%s'%ICFILE )
    output = processInData(nNetwork.model, nNetwork.scaler, testdata)
    output = pd.DataFrame(output)
    if columnNames != []: output.columns = np.append(columnNames, resultTag)
    
    #save output
    nIC = extras.nCardFile(ICFILE)
    S = extras.uploadedFilesFilenameSize
    OCFILE = "OC_%03i_%s"%(nIC, ICFILE[S:])
    
    if columnNames != []: output.to_excel(dir + '/%s'%OCFILE, index=False, header=True)
    else: output.to_excel(dir + '/%s'%OCFILE, index=False, header=False)
	
# PLOTS <<<<<<
    plotPerformance(nNetwork.X_test, nNetwork.Y_test, nNetwork.model)
    plt.savefig(dir + r'/figPerf.png', bbox_inches='tight')
    plt.close()
    
    plotLoss(nNetwork.history)
    plt.savefig(dir + r'/figLoss.png', bbox_inches='tight')
    plt.close()
    
    plotAcc(nNetwork.history)
    plt.savefig(dir + r'/figAcc.png', bbox_inches='tight')
    plt.close()
    
    try:
        plotFeatureImportance(nNetwork.X_train, nNetwork.Y_train)
        plt.savefig(dir + r'/figFeature.png', bbox_inches='tight')
    except:
        pass
    plt.close()
    
    backend.clear_session()
    return OCFILE

def loadModelAndProcess(_dir, ICFILE):
    model = load_model(_dir + r'/model.h5')
    scaler = joblib.load(_dir + r'/scaler.dat')
    
    LCFILE = extras.LearnCardFile(_dir)
    dataset, columnNames = readDataFromFile( _dir + r'/%s'%LCFILE )
    if columnNames != []: resultTag = columnNames[-1]
    else: resultTag = 'prediction'
    
    testdata, columnNames = readDataFromFile( _dir + r'/%s'%ICFILE )
    output = processInData(model, scaler, testdata)
    output = pd.DataFrame(output)
    if columnNames != []: output.columns = np.append(columnNames, resultTag)
    
    backend.clear_session()
        
    #save output
    nIC = extras.nCardFile(ICFILE)
    S = extras.uploadedFilesFilenameSize
    OCFILE = "OC_%03i_%s"%(nIC, ICFILE[S:])
    
    if columnNames != []: output.to_excel( _dir + '/%s'%OCFILE, index=False, header=True)
    else: output.to_excel( _dir + '/%s'%OCFILE, index=False, header=False)
    
    return
    
def plotPerformance(X_test, Y_test, model):
    y_true = Y_test
    Ytoshuffle=np.zeros(len(Y_test))
    for i in range(len(Y_test)):
        Ytoshuffle[i]=Y_test[i]
    np.random.shuffle(Ytoshuffle)
    y_scoresrandom = Ytoshuffle
    print('AUC Random tagger %8.3f \n' % metrics.roc_auc_score(y_true, y_scoresrandom))
    fprr, tprr, thresholdsr = metrics.roc_curve(y_true, y_scoresrandom)
    plt.plot(fprr,tprr, label = 'random')
    y_scores = model.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    plt.plot(fpr,tpr, label = 'NN AUC '+ str(round(metrics.roc_auc_score(y_true, y_scores),2)))
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.yscale('linear')
    plt.legend(loc = 'lower right')
    plt.title('Neural Network performance')
    
    return

def plotLoss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    
    return

def plotAcc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    return

def plotFeatureImportance(X_train, Y_train):
    # Compile Gradient Boosting Regressor
    #gb = GradientBoostingRegressor(n_estimators=100) 
    gb = GradientBoostingClassifier(n_estimators=100) 
    if len(X_train.shape)==3:
        try:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
        except:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
    gb.fit(X_train, Y_train)
    # Show results
    plt.bar(range(X_train.shape[1]), gb.feature_importances_)
    plt.xticks(range(X_train.shape[1]))
    plt.title('Feature importances\n through Gradient Boosting\n')
    plt.xlabel('LearnCard columns')
    plt.ylabel('Relative importance')
    #plt.show()
    #print("Vector of relative importances: "+str(gb.feature_importances_.round(2)))
    return


def test():
    LCFILE = "static/LearnCard_House.xlsx"
    ICFILE = "static/InCard_House.xlsx"
    
    #LCFILE = "util/LearnCard_Increasing.xlsx"
    #ICFILE = "util/InCard_Increasing.xlsx"
    
    LCFILE = "static/LearnCard_Islands.xlsx"
    ICFILE = "static/InCard_Islands.xlsx"
    
    data, columnNames = readDataFromFile(LCFILE)
    if columnNames != []: resultTag = columnNames[-1]
    
    nNetwork = AImodel(data)
    nNetwork.trainModel()
    
    print("=============== PLOT")
    print(nNetwork.X_test)
    print(nNetwork.Y_test)
    plotPerformance(nNetwork.X_test, nNetwork.Y_test, nNetwork.model)
    plt.show()
    
    plotLoss(nNetwork.history)
    plt.show()
    
    plotAcc(nNetwork.history)
    plt.show()
    
    plotFeatureImportance(nNetwork.X_train, nNetwork.Y_train)
    plt.show()
    
    testdata, columnNames = readDataFromFile( ICFILE )
    output = processInData(nNetwork.model, nNetwork.scaler, testdata)
    output = pd.DataFrame(output)
    
    if columnNames != []: output.columns = np.append(columnNames, resultTag)
    
    

    
    
    print(testdata)
    print(output)
    return


if __name__ == '__main__':
    import sys
    test()
    #test("util/LearnCard_Increasing.xlsx")
    #process(str(sys.argv[1]))
