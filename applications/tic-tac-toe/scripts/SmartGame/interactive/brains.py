import numpy as np

import random
import os
import numpy as np
import pandas as pd
import uuid
import time
import pandas as pd
import sys

from sklearn import preprocessing, metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense

from keras import backend
import os
from SmartGame.processing.symmetry_rotation import rotate_to_bottom_left_center_of_mass as rblcm
from SmartGame.generating.classdef import TickTackToe


           

#-----------------------------------------------------------END OF TickTackToe CLASS DEFINITION-------------------------



#-------------------------------------END OF AI FUNCTIONS-----------------------------------------



def tablero_printer(matrix):
    aux = np.asarray(matrix)
    L = len(aux)
    tabla_visual = [[1 for x in range(L)] for x in range(L)] 
    for x in range(L):
        for y in range(L):
            if aux[x][y]==-1: pass
            elif aux[x][y]==1: tabla_visual[x][y] = 'X'
            elif aux[x][y]==0: tabla_visual[x][y] = 'O'
            else: raise Exception('this was not a valid matrix input!')    
    underline = '_'*5*L + '_.'
    filler = '.'*int(((2+5*L)-len('TABLERO ACTUAL'))/2)
    print(f'{filler}Tablero Actual{filler}')
    print(underline)
    for x in range(L):
        q = '|'
        for y in range(L):
            if tabla_visual[x][y]==1: q += '|___|'
            elif tabla_visual[x][y]=='X': q += '|_X_|'
            elif tabla_visual[x][y]=='O': q += '|_O_|'
            else: q = '|err|'
        q += '|'        
        print(q)
    print('\n\n\n')
    return



def request_and_return(matrix):
    aux = np.asarray(matrix)
    L = len(aux)
    tabla_visual = [[1 for x in range(L)] for x in range(L)] 
    for x in range(L):
        for y in range(L):
            if aux[x][y]==-1: pass
            elif aux[x][y]==0: tabla_visual[x][y] = 'O'
            elif aux[x][y]==1: tabla_visual[x][y] = 'X'
            else: raise Exception('this was not a valid matrix input!')    
    underline = '_'*5*L + '_.'
    filler = '.'*int(((2+5*L)-len('TABLERO ACTUAL'))/2)
    print(f'\n\n\n\n\n\n\n{filler}Tablero Actual	{filler}\n')
    print(underline)
    for x in range(L):
        q = '|'
        for y in range(L):
            if tabla_visual[x][y]==1: q += '|___|'
            elif tabla_visual[x][y]=='X': q += '|_X_|'
            elif tabla_visual[x][y]=='O': q += '|_O_|'
            else: q = '|err|'
        q += '|'        
        print(q)
    control = 0
    while control==0:
        print('\n')
        print(f'Donde quieres poner la X? (indices del 1 al {L})')
        print('\nFila:')
        D1 = input()
        if D1.isdigit():
            while not (int(D1) in list(range(1,L+1)) ):
                print(f'\nEntrada Incorrecta... por favor indique un numero del 1 al {L} para la fila\n')
                D1 = input() 
        else:
            print(f'\nEntrada Incorrecta... por favor indique un numero del 1 al {L} para la fila\n')
            D1 = input()
        print('Columna:')
        D2 = input()
        if D2.isdigit():
            while not (int(D2) in list(range(1,L+1))):
                print(f'\nEntrada Incorrecta... por favor indique un numero del 1 al {L} para la columna\n')
                D2 = input()
        else:
            print(f'\nEntrada Incorrecta... por favor indique un numero del 1 al {L} para la columna\n')
            D2 = input()
        D1 = int(D1)-1
        D2 = int(D2)-1
        if matrix.item(D1,D2)==-1: control += 1
        else:
            print('No puede sobreescribir valores; por favor elija de vuelta') 
            control = 0
    #print('\n\n\n')
    print(f'Su respuesta es: FILA {D1+1} y COLUMNA {D2+1} ! \n')
    # pedirle si quiere cambiar antes de salir... 
    return D1,D2

def hacer_y_copiar_desacoplado(A,L,pL):
    B = TickTackToe(L,pL)
    for x in range(L):
        for y in range(L):
            B.board.itemset((x,y),A.board.item(x,y))
    return B



def processInData(model, s, inData):
    inData_scale = s.transform(inData)
    #result = (model.predict(inData_scale) > 0.5).astype("int32") 
    result = model.predict(inData_scale)
    return result

#--------------------------------------------------------------------------
def new_respond(TickTackToe,model,scaler,name,**kwargs):
    L = TickTackToe.length
    pL = TickTackToe.patternLength
    # This is the symmetry rotation________________.
    if True:                                      #|
      orig,_,antirrotation = rblcm(TickTackToe.board)#|
      a = orig.ravel().tolist()[0]                   #|
    else:                                         #|
      #-----o-l-d--v-e-r-s-i-o-n-------------------|
      a = TickTackToe.board.ravel().tolist()[0]
    cases = []
    probas = []
    for K1 in range(L):
        for K2 in range(L):
            if orig.item(K1,K2)==-1:
                copia = hacer_y_copiar_desacoplado(TickTackToe,L,pL)
                copia.board = rblcm(copia.board)[0]
                if kwargs.get('prespective','o')=='o':
                  copia.board.itemset((K1,K2),0)
                else:
                  copia.board.itemset((K1,K2),1)
                b = copia.board.ravel().tolist()[0]
                c = a + b
                c = [int(x) for x in c]
                INPUT = np.asarray(c).reshape(1,-1)          
                result = processInData(model, scaler, INPUT)
                result = result[0][0]
                print(f'result for {K1} and {K2} is ',result)
                cases.append(b)
                probas.append(result)
            else: pass
    print(f'probas are {probas} with antirrotation {antirrotation}')
    move = cases[np.argmax(probas)]
    copia = hacer_y_copiar_desacoplado(TickTackToe,L,pL)
    copia.board = rblcm(copia.board)[0]
    temporal = np.matrix([x for x in np.asarray(move).reshape((L,L)) ])
    print(temporal)
    for j in range(antirrotation):
      temporal = np.rot90(temporal)
    copia.board = temporal
    if kwargs.get('prespective','o')=='o':
      if copia.checkO(): return copia,f"------{name} TE HA GANADO!-------", False 
      else: return copia, f'{name} ha elegido...MIRA:', True       
    elif kwargs.get('prespective','o')=='x':
      if copia.checkX(): return copia,f"------{name} TE HA GANADO!-------", False 
      else: return copia, f'{name} ha elegido...MIRA:', True       

#------------------------------------------------------------------NO MORE TickTackToe FUNCTION


