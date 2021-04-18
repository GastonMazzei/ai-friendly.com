import numpy as np
import pandas as pd


from scipy.stats import bernoulli
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import categorical_crossentropy

from tunnel_generator import *

def balanced_samples(d, partitions=4, **kwargs):
    """
    kwargs: verbose=True
    """
    # Shuffle and select the number of parts it 
    # will be split into
    #
    d = d.sample(frac=1)
    p = 1/partitions
    
    # Define the number of samples per partition:
    # equi-sampling will force us to select the 
    # occurrences of the less abundant
    #
    s = int(d.shape[0]/partitions)
    for a in [x*p for x in range(partitions)]:
        inner_s = sum(d.iloc[:,-1].between(a,a+p))
        if inner_s < s: s = inner_s
    s = int(s)
    if kwargs.get('verbose',True):
        print(f'\nBalancing samples:\n{s} samples per bin with {partitions} bins '\
                f'will transform the {d.shape[0]}-points dataset\ninto a '\
                f'{partitions*s}-points dataset!')
    
    # retrieve 's' items per class
    #
    data = []
    for a in [x*p for x in range(partitions)]:
        data += [d[d.iloc[:,-1].apply(lambda x: round(x,1)).between(a,a+p)].sample(frac=1).iloc[:s,:]]
    if kwargs.get('verbose',True):
        print('\nDataset balanced successfully!\n')
    return pd.concat(data,0)


def main(required_length = 15E3, vainilla = True, size=300, partition=50):
    df = pd.DataFrame()
    count = 0
    q = 0
    LIMIT, INNER_LIMIT = 25, 5
    while ((len(df)<required_length) and (count<LIMIT)):
        print(f'\nIteration {count}:\nRequired Length: {required_length}\n'\
              f'Current Length: {len(df)}\n')
        if vainilla:
          df = generator(
                          50+q,
                          50+q,
                          50+q,
                          verbose=True,
                          domain = [np.logspace(-3,3,50+q),
                                    np.logspace(-2,0,50+q)-0.001,
                                    np.logspace(-2,2,50+q)]
                                      ,)
                                           
          if (len(df)>=required_length) or (counter%INNER_LIMIT==0): 
            df = balanced_samples(df, partition, verbose=True)
          q += 10

        else: 
          df = generator(
                            int(1+q/5),
                            int(1+q/5),
                            int(1+q/5),
                            verbose=False,
                            vainilla=False,
                            size=size,)
          q += 5

        count += 1
    if len(df)>=required_length: print('\nSUCCESS! the required-length-condition WAS SATISFIED')
    else: print('\nFAILURE: the required-length-condition WAS NOT SATISFIED')
    print(f'\n\nREQUIRED: {required_length}\nACTUAL: {len(df)}')
    return df

