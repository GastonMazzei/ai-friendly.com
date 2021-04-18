# Imports clasicos
import sys
import os
import matplotlib.pyplot as plt
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

from PLOTTERS import plotLoss, plotAcc, plotPerformance
from AIMODEL import AI_model
#------BEGIN SCRIPT!

def main():    
    data_directorio = 'database'#os.getcwd()
    print_directorio = 'gallery'
    df = pd.read_csv(data_directorio+'/database.csv')
    dataset = df.to_numpy()
    #                     custom_model=False
    nNetwork = AI_model(dataset,False)
    nNetwork.trainModel()
    #nNetwork.model.save(data_directorio + '/model.h5')
    #joblib.dump(nNetwork.scaler, data_directorio + '/scaler.dat') 
    result = {}
    result['confidence'] = nNetwork.confidence
    # PLOTTING:
    plotPerformance(nNetwork.X_test, nNetwork.Y_test, nNetwork.model)
    plt.savefig(print_directorio + '/figPerf.png', bbox_inches='tight')
    plt.close()
    plotLoss(nNetwork.history)
    plt.savefig(print_directorio + '/figLoss.png', bbox_inches='tight')
    plt.close()
    plotAcc(nNetwork.history)
    plt.savefig(print_directorio + '/figAcc.png', bbox_inches='tight')
    plt.close()
    backend.clear_session()
    return print('SUCCESS!')

    
def process(dir):
    print(' ========== AI PROCESS ========== ')
    LCFILE = extras.LearnCardFile(dir)
    print(f'\n\nGONNA READ THIS: {LCFILE}\n\n') 
    dataset, columnNames = readDataFromFile( dir+'/'+ LCFILE )
    if columnNames != []: resultTag = columnNames[-1]
    else: resultTag = 'prediction'
    with open(dir+"/architecture.txt", "r") as file:
        model = eval(file.readline())
    print('\n\n','DEBUG: READ THE TXT AND IT IS',model,'\n\n')
    nNetwork = AImodel(dataset,model)
    nNetwork.trainModel()
    nNetwork.model.save(dir + r'/model.h5')
    joblib.dump(nNetwork.scaler, dir + r'/scaler.dat') 
    result = {}
    result['confidence'] = nNetwork.confidence
    ICFILE = extras.InCardsFiles(dir)[0]
    testdata, columnNames = readDataFromFile( dir + r'/%s'%ICFILE )
    output = processInData(nNetwork.model, nNetwork.scaler, testdata)
    output = pd.DataFrame(output)
    if columnNames != []: output.columns = np.append(columnNames, resultTag)
    nIC = extras.nCardFile(ICFILE)
    S = extras.uploadedFilesFilenameSize
    OCFILE = "OC_%03i_%s"%(nIC, ICFILE[S:])
    if columnNames != []: output.to_excel(dir + '/%s'%OCFILE, index=False, header=True)
    else: output.to_excel(dir + '/%s'%OCFILE, index=False, header=False)
    plotPerformance(nNetwork.X_test, nNetwork.Y_test, nNetwork.model)
    plt.savefig(dir + r'/figPerf.png', bbox_inches='tight')
    plt.close()
    plotLoss(nNetwork.history)
    plt.savefig(dir + r'/figLoss.png', bbox_inches='tight')
    plt.close()
    plotAcc(nNetwork.history)
    plt.savefig(dir + r'/figAcc.png', bbox_inches='tight')
    plt.close()
    backend.clear_session()
    return OCFILE

main()










