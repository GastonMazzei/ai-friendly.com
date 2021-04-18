import numpy as np
import pandas as pd

from scipy.stats import bernoulli
from sklearn.preprocessing import StandardScaler

from functools import partial

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import categorical_crossentropy

def transmission_coefficient(L,V,E):
    mass_by_hbar_squared = 1
    squared_gamma_by_two = ((1-E/V)/(E/V)+(E/V)/(1-E/V)-2)/4
    beta = np.sqrt(2*mass_by_hbar_squared*(V-E))
    T_inverse = np.cosh(beta*L)**2 + squared_gamma_by_two * np.sinh(beta*L)**2
    return T_inverse**(-1)
    
def transmission_dispatcher(L,V,E,**kwargs):
    T = transmission_coefficient(L,V,E)
    if kwargs.get('categorical',True):
        return bernoulli.rvs(T,0,kwargs.get('size',1000))
    else:
        return T


def generator(Ln=50, En=50, Vn=50, **kwargs):
    """
    kwargs: 
        verbose=True
        vainilla=True
            True = each iteration adds (L,V,E,T)
            False = each iteration adds 'size' number
                    of vectors like this (e.g. T=0.25, size=4):
                            L V E 0
                            L V E 0
                            L V E 1
                            L V E 0
            domain = [[Lv],[Vv],[Ev]], default=False
            size = 1000
            size for vainilla=False case
    """
    LIM = 100E3
    def vainilla(lv,v,ev):
        nonlocal df
        df['L/V'].append(lv)
        df['V'].append(v)
        df['E/V'].append(ev)
        df['T'].append(
            transmission_dispatcher(
                lv*v,v,ev*v,categorical=False))
        return df
    #
    def categorical(size,L,V,E):
        nonlocal df
        df['L'] += [L] * size
        df['V'] += [V] * size
        df['E'] += [E] * size
        df['proba'] += list(
            transmission_dispatcher(
                L,V,E,categorical=True, size=size))        
        return df

    if kwargs.get('vainilla', True):
        main = vainilla
    else:
        main = partial(categorical, kwargs.get('size',1000))
    if kwargs.get('domain',False):
        L_V, E_V, V = kwargs['domain']
    else:
        L_range = np.linspace(1E-4,1E1,Ln)
        E_range = np.linspace(1E-1,1E1,En)
        V_range = np.linspace(1E-1,5E1,Vn)
        if kwargs.get('verbose',True):
            print(f'\nGenerating {Ln*En*Vn} simulations...')
    #
    df = {'L/V':[],
          'V':[],
          'E/V':[],
          'T':[],
         }
    for lv in L_V:
        for v in V:
            for ev in [x for x in E_V if x<v]:
              main(lv,v,ev)
    #
    if kwargs.get('verbose',True):
        print(f'\nData generation ended successfully!')
        print(f'\nTotal: {len(df["T"])} cases made sense and were saved!')
    for x in df.keys(): print(len(df[x]))
    return pd.DataFrame(df)

