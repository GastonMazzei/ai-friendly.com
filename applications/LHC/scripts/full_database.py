import pandas as pd
import numpy as np
from itertools import combinations
from os import listdir
import os
import sys

def make_dataset(p_train, p_val, p_test, cuadrinorm = False,**kwargs):
 
    # (1) Define random seed and dir
    seed = kwargs.get('seed',93650115)
    dire = kwargs.get('dire',False)

    # (2) Define useful functions
    #
    # (2.a) make the cuadrinorm of P_1+P_2 
    #       con signature (-,+,+,+) 
    def spit_cuadrinorm(dft):
        return (-(dft[0]+dft[4])**2 + (dft[1]+dft[5])**2 
               + (dft[2]+dft[6])**2 + (dft[3]+dft[7])**2)

    # (2.b) mix signal with background
    #       under "key" (train,val,test)
    #       default values and "q" mix ratio 
    def mix_p_concentrated(data,key,q,**kwargs):
        # A,signal B,background
        A = data[0] 
        B = data[1] 
        # filtering constraint
        if len(A)==len(B):
            L = len(A)
        else:
            print('script not ready for different-sized'\
                  ' signals and background') 
            raise ValueError
        # def default values as a function of key
        if key=='train': Ln = kwargs.get(
                           'trainingset_length',
                                          #28000,)
                                          13300,)
        elif key=='test': Ln = kwargs.get(
                            'testingset_length',
                                           3000,)
        else: Ln = kwargs.get(
                         'validationset_length',
                                           3000,)
        # main          
        C = pd.concat([A[:int(Ln*q)], B[:int(Ln*(1-q))]])
        cols = tuple(C.columns)
        for x_ in range(len(cols)//11):
            try:
                C[cols[x_*11]] = x_ + 1
            except: pass
        # change np.non to zeros
        C = C.fillna(0)
        if seed: return C#.sample(frac=1, random_state=seed)
        else: return C.sample(frac=1)

    # (2.c) well-behaved saver: don't 
    #       overwrite if it exists
    def saver(df,tag,optionaldir=False):
        if optionaldir:
            base = os.getcwd() 
            os.chdir(optionaldir)
        if tag+'.csv' in listdir(): 
            print('WARNING: file already existed (no-overwrite-policy)')
        else:  
            df.to_csv(tag+'.csv',index=False,header=False)        
            print(f'STATUS: --((successfully saved {tag}))--')
        if optionaldir: os.chdir(base)
        return
    # (3) Load the Data
    if seed:
        signal = pd.read_csv('database/signal.csv', low_memory=False)#.sample(frac=1,random_state=seed)
        background = pd.read_csv('database/background.csv', low_memory=False)#.sample(frac=1, random_state=seed)
    else:
        signal = pd.read_csv('database/signal.csv', low_memory=False).sample(frac=1)
        background = pd.read_csv('database/background.csv', low_memory=False).sample(frac=1)

    #
    # (4) Last col (0-1) tagging and
    #     Optional Cuadrinorm Column
    t_ = max([len(signal.columns),len(background.columns)])
    print('using ',t_,' instead of "8"')
    if cuadrinorm:
        signal[t_] = spit_cuadrinorm(signal)
        background[t_] = spit_cuadrinorm(background)
        t_ += 1
    signal[t_] = 1
    background[t_] = 0
    # (5) Mix signal and background under "p_train", "p_val",
    #     "p_test", "kwargs" specs
    if True:
        training_set_raw = [signal[:14000], background[:14000]] #[signal.iloc[:14000,:], background.iloc[:14000,:]]
        training_set = mix_p_concentrated(training_set_raw, 'train', p_train,**kwargs)
    validation_set_raw = [signal[-6000:-3000], background[-6000:-3000]]
    testing_set_raw = [signal[-3000:], background[-3000:]]
    validation_set = mix_p_concentrated(validation_set_raw, 'valid', p_val,**kwargs) 
    testing_set = mix_p_concentrated(testing_set_raw, 'test', p_test,**kwargs)

    # (6) CORRECTNESS TESTS:
    if True:
        V = [training_set, validation_set, testing_set] 
        #(6.1): Uniqueness Test
        uniqueness_test(*V)     
        # (6.2): Tolerance Test
        tolerance_test(*V,p_train,p_val,p_test) 
        # (6.3): Length Test
        length_test(*V) 

    # (7) SAVE
    saver(training_set,f'train-{int(100*p_train)}',dire)
    saver(validation_set,f'val-{int(100*p_val)}',dire)
    saver(testing_set,f'test-{int(100*p_test)}',dire)    
    print(f'STATUS: SUCCESS w params p {p_train} {p_val} {p_test}, seed = {seed} , cuadrinorm = {cuadrinorm}')
    return 

def uniqueness_test(a,b,c):
    """
    Check that test,train,val have
    different datapoints under a 
    certain threshold (1%)
    """
    threshold = 0.01
    for X in combinations([a,b,c],2):
        if sum(pd.concat([X[0],X[1]]).duplicated())/(len(X[0])+len(X[1]))>threshold:
            raise Exception('\n\nBuildingError: uniqueness test failed\n\n')
        else: pass
    print('STATUS: Uniqueness Test Passed!')
    return

def tolerance_test(a,b,c,p1,p2,p3):
    """
    Check that the tagged_proba
    has the correct signal_noise_ratio
    under a certain threshold (1%?)
    """
    # threshold understood as.....
    # deviation from the fraction
    # e.g. 1% means 0.01*0.99 signal 
    # will be tolerated, NOT 0.01+0.01
    threshold = 0.01
    x = [a,b,c]
    y = [p1,p2,p3]
    for i in range(3):
        if (y[i]*(1+threshold) > sum(x[i].iloc[:,-1])/len(x[i]) and
            sum(x[i].iloc[:,-1])/len(x[i]) > y[i]*(1-threshold) ): pass
        else:
          print(y[i]*(1+threshold)) #
          print(sum(x[i].iloc[:,-1])/len(x[i])) #
          print('index was: ',i)  #
          raise Exception('\n\nBuildingError: tolerance test failed\n\n')         
    print('STATUS: Tolerance Test Passed!')
    return

def length_test(a,b,c):
    """
    Check that train,test,val
    have all the required length 
    i.e. 
    TRAIN = 28k
    VAL = 3k
    TEST = 3k
    """
    if (len(a)==13300 and
        len(b)==3000  and
        len(c)==3000): pass
    else: raise Exception("\n\nBuildingError: length test failed\n\n")
    print('STATUS: Length Test Passed!')
    return


if __name__=='__main__':
    #make_dataset(.02,.02,.02,False,dire='temp')
    make_dataset(.5,.5,.5,False,dire='database')










