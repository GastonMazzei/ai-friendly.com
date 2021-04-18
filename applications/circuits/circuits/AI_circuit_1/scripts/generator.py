import random

import matplotlib.pyplot as plt
import pandas as pd
import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()

from pathlib import Path
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import SubCircuitFactory
from PySpice.Spice.Parser import SpiceParser
from PySpice.Unit import *
from math import ceil

from matplotlib.pyplot import imshow

from PySpice.Spice.Netlist import Circuit
import numpy as np

import sys



libraries_path = '../models'
spice_library = SpiceLibrary(libraries_path)
 

def simulate_RLC(w=2000,V=2,R=10,L=3,C=320,**kwargs):

    circuit = Circuit('Name me please')
    circuit.R('1',  1, 2,R@u_kΩ)
    circuit.L('1',  2, 3,L@u_H)
    circuit.C('1',  3, 0,C@u_nF)
    #circuit.V('1',  circuit.gnd,1 ,f'SIN(0 {V} {w})')
    circuit.V('1',0,1, f'DC 0 AC {V} SIN(0 {V} {w})')
    dt = 2*np.pi/w*4
    tf = 5/w
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    #analysis = simulator.transient(step_time=0.1@u_ms, end_time=tf@u_s)
    analysis = simulator.transient(step_time=dt@u_ms, end_time=tf@u_s)

    if kwargs.get('view',False):
        #print(str(circuit))
        fig = plt.figure(figsize=(20,4))  # create a figure object
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(analysis['1']-analysis['2'],label='R')
        plt.plot(analysis['2']-analysis['3'],label='L')
        ax.plot(analysis['3'],label='C')
        #ax.set_ylim(-int(V*1.1)-10,int(V*1.1)+10)
        ax.legend()
        ax.set_title(f'freq : {w}, voltage: {V}')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('current (ma)')
        plt.savefig('gallery/RLC_example.png')
        print(f'resonance was at {np.sqrt(1/(L*C))}')
    return max(abs(analysis['1']-analysis['2']))/R

def measure_resonance(C_values: list=[10,50,250,], L_values: list=[0.3,2]):
  fig, ax = plt.subplots(figsize=(20,6))
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  if len(colors)<len(C_values)*len(L_values):
    colors *= ceil(len(colors)/(len(C_values)*len(L_values)))
  counter = 0
  for C in C_values:
      for L in L_values:
          f_zero = np.sqrt(1E9)/np.sqrt(L*C)/2/np.pi
          x=np.logspace(0,4,100)
          y = []
          R = .5#.1 default
          alpha = R / L * 1000
          for j in x:
              y.append(float(simulate_RLC(j,2,R,L,C, view=False)))
          #ax.axvspan(f_zero-alpha/2,f_zero+alpha/2,alpha=0.1,color=colors[counter])
          ax.scatter(x,y,label=f'f={int(f_zero)}',c=colors[counter])
          ax.axvline(linewidth=0.7, x=f_zero, color=colors[counter])
          ax.plot(x,y)
          counter+=1
  ax.set_xlabel('frequency')
  ax.set_ylabel('current (mA)')
  ax.legend()
  ax.set_xscale('log')
  ax.set_xlim([100,8000])
  plt.savefig('gallery/RLC_resonance-curves.png')

def make_database_core(w=2000,V=12,R=10,L=3,C=320):
    threshold = 4
    circuit = Circuit('RLC-series')
    circuit.V('1',0,1, f'DC 0 AC {V} SIN(0 {V} {w})')
    circuit.R('1',  1, 2,R@u_kΩ)
    circuit.L('1',  2, 3,L@u_H)
    circuit.X('diodus','D1N4148', 2, 3)
    circuit.include(spice_library['D1N4148']) 
    circuit.C('1',  3, 0,C@u_nF)
    dt = 2*np.pi/w*4
    tau = 2*np.pi/w*4
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=dt@u_ms, end_time=tau@u_s)
    auxiliar = analysis['2']-analysis['3']
    return [w,V,R,L,C,int(float(max(auxiliar))>threshold)]

def make_database_iterator(DATABASE_SIZE=5000):
  W_vector = np.logspace(2,5,50)
  V_vector = np.linspace(0.15,15,50)
  R_vector = np.linspace(1/20,20,50)
  L_vector = np.linspace(1/100,10,50)
  C_vector = [x/100 for x in range(1,1000,50)]   
  answ = []
  counter = 0
  already_used = []

  def choose():
    temp_V = random.choice(V_vector)
    temp_R = random.choice(R_vector) 
    temp_L = random.choice(L_vector) 
    temp_C = random.choice(C_vector)
    temp_W = random.choice(W_vector)
    return temp_W, temp_V, temp_R, temp_L, temp_C
  def repeating_filter(l: list):
    if tuple(l) in already_used:
      repeating_filter(choose())
    else:
      already_used.append(tuple(l))
    return l
  while counter <DATABASE_SIZE:
      answ += [make_database_core(*repeating_filter(choose()))]
      counter += 1
  data = pd.DataFrame(answ,columns=['W','V','R','L','C','D'])
  data.to_csv("database/database.csv",index=False)

if __name__=='__main__':
  if sys.argv[1]=='1':
    simulate_RLC(view=True)
  elif sys.argv[1]=='2':
    measure_resonance()
  elif sys.argv[1]=='3':
    print('activated 3')
    make_database_iterator(5000)
