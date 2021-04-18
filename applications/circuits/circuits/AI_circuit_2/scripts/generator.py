import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import SubCircuitFactory
from PySpice.Spice.Parser import SpiceParser
from PySpice.Unit import *

from math import sqrt 
from matplotlib.pyplot import imshow
from PySpice.Spice.Netlist import Circuit

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

libraries_path = '../models'
spice_library = SpiceLibrary(libraries_path)

def simulate_sallen_key_low_pass_filter():
  # Initialize the Circuit
  circuit = Circuit('Sallen-key low-pass-filter')
  # Include the active element: operational amplifier
  circuit.include(spice_library['LM741'])
  V = 2
  w = 2E3
  circuit.V('1', 1,circuit.gnd,f'DC 0 AC {V} SIN(0 {V} {w})')
  R_1 = 11.2
  R_2 = 11.2
  R_A = 1000
  R_B = 0.01
  C_1 = 200
  C_2 = 100
  links = [(1,2),
          (2,3),
          (circuit.gnd,4),
          (4,5),]
  R_vector = [R_1,R_2,R_A,R_B,]
  for x in range(len(R_vector)):
    circuit.R(str(x+1),  links[x][0], links[x][1],
                                    R_vector[x]@u_kΩ)
  circuit.C('1',  2, 5,C_1@u_pF)
  circuit.C('2',  circuit.gnd, 3,C_2@u_pF)
  circuit.X('opamp','LM741',3,4,'Vcc','Vee',5)
  circuit.V('2','Vcc',circuit.gnd,'DC +15')
  circuit.V('3','Vee',circuit.gnd,'DC -15')
  f = w
  simulator = circuit.simulator(temperature=25, nominal_temperature=25)
  analysis = simulator.transient(step_time=((1/f)/10)@u_s, end_time=(4/f)@u_s)
  fig, ax = plt.subplots(figsize=(20,5))
  ax.plot(analysis['5'],label='out')
  ax.plot(analysis['1'],label='in')
  ax.legend()
  plt.savefig('gallery/sallen-key-low-pass-filter_example.png')

def simulate_attenuation_factor_core(w,R_1,R_2,C_1,C_2,optboolean=False):
    circuit = Circuit('Name me please')
    circuit.include(spice_library['D1N4148']) 
    # TLV3201 is a 0-5 T.I. OpAmp
    circuit.include(spice_library['LM741'])
    V = 7
    #w = 20E3
    #circuit.V('1', circuit.gnd,1,f'DC 0 AC {V} SIN(0 {V} {w})')
    circuit.V('1', 1,circuit.gnd,f'DC 0 AC {V} SIN(0 {V} {w})')
    #R_1 = 11.2
    #R_2 = 11.2
    R_A = 100000
    R_B = 0.001
    #C_1 = 2000
    #C_2 = 1000
    links = [(1,2),
            (2,3),
            (circuit.gnd,4),
            (4,5),]
    R_vector = [R_1,R_2,R_A,R_B,]
    for x in range(len(R_vector)):
        circuit.R(str(x+1),  links[x][0], links[x][1],
                                      R_vector[x]@u_kΩ)
    circuit.C('1',  2, 5,C_1@u_pF)
    circuit.C('2',  circuit.gnd, 3,C_2@u_pF)
    circuit.X('opamp','LM741',3,4,'Vcc','Vee',5)
    circuit.V('2','Vcc',circuit.gnd,'DC +15')
    circuit.V('3','Vee',circuit.gnd,'DC -15')
    if optboolean:
        circuit.X('diodus','D1N4148', 2,1)
    f = w
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=((1/f)/10)@u_s, end_time=(4/f)@u_s)    # Printer?
    if False:
        #print(str(circuit))
        fig = plt.figure(figsize=(20,4))  # create a figure object
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(analysis['1']-analysis['2'],label='R')
        plt.plot(analysis['2']-analysis['3'],label='L')
        ax.plot(analysis['3'],label='C')
        #ax.set_ylim(-int(V*1.1)-10,int(V*1.1)+10)
        ax.legend()
        ax.set_title(f'freq : {w}')
        print(f'resonance was at {np.sqrt(1/(L*C))}')
    if True:
        outputter = max(analysis['5'])
        del(circuit)
        return outputter
    return

def simulate_attenuation_factor():
    fig, ax = plt.subplots(figsize=(20,6))
    for C_1 in [1000,2000,3000]:
        for C_2 in [1500,2000]:
            R_1 = 11.2
            R_2 = 11.2
            f_zero = 1/sqrt((C_1/1E3)*(C_2/1E3)*R_1*R_2*1E-12)/2/np.pi
            #print(f'f_zero es {int(f_zero)}')
            try: 
                x = [i for i in range(1,100000,500)]
            except: break
            y = []
            #alpha = R / L * 1000
            for j in x:
                y.append(float(simulate_attenuation_factor_core(j,R_1,R_2,C_1,C_2)))
            #ax.axvspan(f_zero-alpha/2,f_zero+alpha/2,alpha=0.2)
            ax.scatter(x,y,label=f'f={round(f_zero,-3)}')
            ax.plot(x,y)
            col = plt.gca().lines[-1].get_color()
            ax.axvline(linewidth=1.5,color=col, x=f_zero)
    ax.set_xlabel('freq (Hz)')
    ax.set_ylabel('Vpeak (V)')
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim([500,100000])
    plt.savefig('gallery/sallen-key-low-pass-filter_attenuation_factor.png')

def make_database_core(w,V,R_1,R_2,C_1,C_2,ax=False):
    circuit = Circuit('Sallen-key low-pass-filter')
    circuit.include(spice_library['D1N4148']) 
    circuit.include(spice_library['LM741'])
    circuit.V('1', 1,circuit.gnd,f'DC 0 AC {V} SIN(0 {V} {w})')
    R_A = 100000
    R_B = 0.001
    links = [(1,2),
             (2,3),
             (circuit.gnd,4),
             (4,5),]
    R_vector = [R_1,R_2,R_A,R_B,]
    for x in range(len(R_vector)):
        circuit.R(str(x+1),  links[x][0], links[x][1],
                                       R_vector[x]@u_kΩ)
    circuit.C('1',  2, 5,C_1@u_pF)
    circuit.C('2',  circuit.gnd, 3,C_2@u_pF)
    circuit.X('opamp','LM741',3,4,'Vcc','Vee',5)
    circuit.V('2','Vcc',circuit.gnd,'DC +15')
    circuit.V('3','Vee',circuit.gnd,'DC -15')
    circuit.X('diodus','D1N4148', 2,1)
    f = w
    tf = 3/w
    t0 = 1
    treshold = 4
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=((1/f)/10)@u_s, end_time=3/w@u_s)
    if False:
        ax.plot(analysis['5'],label='output')
        ax.plot(analysis['1'],label='input')
        ax.plot(analysis['1']-analysis['2'],label='diodo')
        ax.hlines(4,0,5/w)
        ax.legend()
        ax.set_title(f'R1: {R_1}, R2: {R_2}, C1: {C_1}, C2: {C_2}, V: {V}, W: {w}')
        ax.set_ylim(-15,15)
    auxiliar = analysis['2']-analysis['1']
    del(circuit)        
    return [w,V,R_1,R_2,C_1,C_2,int(float(max(auxiliar))>treshold)]    

def make_database_iterator():
    W_vector = np.logspace(0,5,2000) #[int(x*100) for x in range(1,3000)]
    V_vector = np.logspace(-2,3,2000) #[x/10 for x in range(1,150)]
    R1_vector = np.logspace(-2,5,2000) #[x/20 for x in range(1,51,1)]
    R2_vector = np.logspace(-2,5,2000) #[x/20 for x in range(1,51,1)]
    C1_vector = np.logspace(-2,4,2000) #[int(x*100) for x in range(1,100,2)]
    C2_vector = np.logspace(-2,4,2000) #[int(x*100) for x in range(1,100,2)]
    answ = []
    times = 10000
    counter = 0
    errorlog = []
    while counter <times:
        try: 
            answ += [make_database_core(random.choice(W_vector),
        random.choice(V_vector),
        random.choice(R1_vector),
        random.choice(R2_vector),
        random.choice(C1_vector),
        random.choice(C2_vector),
                      )]
        except Exception as ins:
            errorlog.append(ins.args)
        counter += 1

    data = pd.DataFrame(answ,columns=['W','V','R1','R2','C1','C2','D'])
    data.to_csv('database/database.csv',index=False)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='1':
        simulate_sallen_key_low_pass_filter()
    elif sys.argv[1]=='2':
        simulate_attenuation_factor()
    elif sys.argv[1]=='3':
        make_database_iterator()

