import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Imports de estadistica
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split

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
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    return

