
import pandas as pd
import numpy as np

from numpy.random import rand
from matplotlib import pyplot as plt
from extras import *

for tt in range(nt):
    field = rand()*(-1)**np.random.randint(2)
    E1 = M1 = E2 = M2 = 0
    config = initialstate(N)
    iT=1.0/T[tt]; iT2=iT*iT;
    
    for i in range(eqSteps):         
        mcmove(config, iT,field)           # Monte Carlo moves

    for i in range(mcSteps):
        mcmove(config, iT,field)           
        Ene = calcEnergy(config, field)     # calculate the energy
        Mag = calcMag(config)        # calculate the magnetisation

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag 
        E2 = E2 + Ene*Ene

    H[tt] = float(field)
    E[tt] = n1*E1
    M[tt] = n1*M1
    C[tt] = (n1*E2 - n2*E1*E1)*iT2
    X[tt] = (n1*M2 - n2*M1*M1)*iT

if False:
  f = plt.figure(figsize=(18, 10)); # plot the calculated values    
  sp =  f.add_subplot(2, 2, 1 );
  plt.scatter(T, E, s=50, marker='o', color='IndianRed')
  plt.xlabel("Temperature (T)", fontsize=20);
  plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');
  sp =  f.add_subplot(2, 2, 2 );
  plt.scatter(T, abs(M), s=50, marker='o', color='RoyalBlue')
  plt.xlabel("Temperature (T)", fontsize=20); 
  plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');
  sp =  f.add_subplot(2, 2, 3 );
  plt.scatter(T, C, s=50, marker='o', color='IndianRed')
  plt.xlabel("Temperature (T)", fontsize=20);  
  plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   
  sp =  f.add_subplot(2, 2, 4 );
  plt.scatter(T, X, s=50, marker='o', color='RoyalBlue')
  plt.xlabel("Temperature (T)", fontsize=20); 
  plt.ylabel("Susceptibility", fontsize=20);   plt.axis('tight');
  plt.show()

df = pd.DataFrame({'field':H ,'T':T,'E':E,'C':C,'X':X,'M':M}) # append
with open('datasets/ising.csv', 'a') as f:
    df.to_csv(f, mode='a', header=f.tell()==0, index=False)

#https://rajeshrinet.github.io/blog/2014/ising-model/

