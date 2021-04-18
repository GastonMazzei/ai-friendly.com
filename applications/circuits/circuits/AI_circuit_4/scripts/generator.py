import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from math import sqrt 

import random
import itertools

from pathlib import Path
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import SubCircuitFactory
from PySpice.Spice.Parser import SpiceParser
from PySpice.Unit import *
from PySpice.Spice.Netlist import Circuit

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

libraries_path = '../models'
spice_library = SpiceLibrary(libraries_path)


def simulate_amplifier():
    circuit = Circuit('Amplifier')
    R_1 = [100]
    R_2 = [20]
    R_C = [10]
    R_E = [2]
    R_L = [1]
    C_1 = [10]
    C_2 = [10]
    R1 = random.choice(R_1)
    R2 = random.choice(R_2)
    RC = random.choice(R_C)
    RE = random.choice(R_E)
    RL = random.choice(R_L)
    C1 = random.choice(C_1)
    C2 = random.choice(C_2)
    circuit.R(1, 5, 2, R1@u_kΩ) #kOhm 
    circuit.R(2, 2, 0, R2@u_kΩ)  #kOhm
    circuit.R('C', 5, 4, RC@u_kΩ) #kOhm
    circuit.R('E', 3, 0, RE@u_kΩ)  #kOhm
    circuit.R('Load', 'out', 0, RL@u_MΩ)  #MOhm
    circuit.C(1, 'in', 2, C1@u_uF)    #uF
    circuit.C(2, 4, 'out', C2@u_uF)   #uF
    circuit.BJT(1, 4, 2, 3, model='bjt') # Q is mapped to BJT !
    circuit.model('bjt', 'npn', bf=80, cjc=pico(5), rb=100)
    V_vector = [0.5]
    w_vector = [1E3]
    V = random.choice(V_vector)
    w = random.choice(w_vector)
    circuit.V('power', 5, circuit.gnd, 15@u_V)
    circuit.V('var','in',circuit.gnd, f'DC 0 AC {V} SIN(0 {V}V {w})')
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time = (1/w/25)@u_s,end_time=(4/w)@u_s)
    figure = plt.figure(1, (20, 10))
    axe = plt.subplot(111)
    plt.title('')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.grid()
    plot(analysis['in'], axis=axe)
    plot(analysis.out, axis=axe)
    plt.legend(('input', 'output'), loc=(.05,.1))
    plt.tight_layout()
    plt.savefig('gallery/amplifier.png')
    return

def simulate_amplification_factor_core(R1,R2,RC,RE,RL,C1,C2,V,w):
    circuit = Circuit('Amplifier')
    circuit.R(1, 5, 2, R1@u_kΩ) #kOhm 
    circuit.R(2, 2, 0, R2@u_kΩ)  #kOhm
    circuit.R('C', 5, 4, RC@u_kΩ) #kOhm
    circuit.R('E', 3, 0, RE@u_kΩ)  #kOhm
    circuit.R('Load', 'out', 0, RL@u_MΩ)  #MOhm
    circuit.C(1, 'inp', 2, C1@u_uF)    #uF
    circuit.C(2, 4, 'out', C2@u_uF)   #uF
    circuit.BJT(1, 4, 2, 3, model='bjt') # Q is mapped to BJT !
    circuit.model('bjt', 'npn', bf=80, cjc=pico(5), rb=100)
    circuit.V('power', 5, circuit.gnd, 15@u_V)
    circuit.V('var','inp',circuit.gnd, f'DC 0 AC {V} SIN(0 {V}V {w})')
    T = 25
    simulator = circuit.simulator(temperature=T, nominal_temperature=T)
    analysis = simulator.transient(step_time = (2*np.pi/w/20)@u_s,end_time=(2*np.pi/w*4)@u_s)
    return [float(min(analysis['out']))]

def simulate_amplification_factor_iterator():
    R_1 = [100,50,20,200,150]
    R_2 = [20,30,40,10]
    R_C = [10,7,5,3,20,25,30]
    RC = [5]
    R_E = [2,3,5,8,4]
    R_L = [1,2,3,0.5,0.2]
    C_1 = [10,15,20,5]
    C_2 = [10,15,30,40,5]
    V_vector = [0.5,1.3,0.8,3,2.5]
    w_vector = [1E3]
    R1 = random.choice(R_1)
    R2 = random.choice(R_2)
    RC = random.choice(R_C)
    RE = random.choice(R_E)
    RL = random.choice(R_L)
    C1 = random.choice(C_1)
    C2 = random.choice(C_2)
    V = random.choice(V_vector)

    f,ax = plt.subplots(figsize=(25,9))
    w_iter = [np.exp(x/15) for x in range(1,150)]
    Ntot=9
    y = [[] for x in range(Ntot)]
    for q in range(Ntot):    
        for w in w_iter:
            y[q] += simulate_amplification_factor_core(R1,R2,RC,RE,RL,C1,C2,V,w)
        y[q] = [-x/V for x in y[q]]        
        ax.scatter(w_iter,y[q])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Vout/Vin')
    ax.set_xlabel('Freq (Hz)')
    plt.savefig('gallery/amplification-factor_vs_frequency.png')



def make_database_core(R1,R2,RC,RE,RL,C1,C2,V,w,l=False):
    circuit = Circuit('Amplifier')
    circuit.R(1, 5, 2, R1@u_kΩ) #kOhm 
    circuit.R(2, 2, 0, R2@u_kΩ)  #kOhm
    circuit.R('C', 5, 4, RC@u_kΩ) #kOhm
    circuit.R('E', 3, 0, RE@u_kΩ)  #kOhm
    circuit.R('Load', 'out', 0, RL@u_MΩ)  #MOhm
    circuit.C(1, 'inp', 2, C1@u_uF)    #uF
    circuit.C(2, 4, 'out', C2@u_uF)   #uF
    circuit.BJT(1, 4, 2, 3, model='bjt') # Q mapped to BJT 
    circuit.model('bjt', 'npn', bf=80, cjc=pico(5), rb=100)
    circuit.V('power', 5, circuit.gnd, 15@u_V)
    circuit.V('var','inp',circuit.gnd, f'DC 0 AC {V} SIN(0 {V}V {w})')
    circuit.include(spice_library['D1N4148']) 
    circuit.X('diode','D1N4148',3,2)
    T = 25
    simulator = circuit.simulator(temperature=T, nominal_temperature=T)
    analysis = simulator.transient(step_time = (1/w/10)@u_s,end_time=(2/w)@u_s)
    treshold = 4
    return [R1,R2,RC,RE,RL,C1,C2,V,w,
            int(treshold<float(max(analysis['3']-analysis['2'])))]

    
def make_database():
    R_1 = np.logspace(-3, 4, 2000) #[x/10 for x in range(1,1000)]
    R_2 = np.logspace(-3, 4, 2000) #[x/10 for x in range(1,1000)]
    R_C = np.logspace(-3, 4, 2000) #[x/20 for x in range(1,1000)]
    R_E = np.logspace(-3, 4, 2000) #[x/20 for x in range(1,1000)]
    R_L = np.logspace(-1,1, 2000) #[x/20 for x in range(1,2000)] # ESTA EN MEGOHMS.. QUE SEA SOLO ID=1! 
    C_1 = np.logspace(-3, 3, 2000) #[x/50 for x in range(1,2000,10)]
    C_2 = np.logspace(-3, 3, 2000) #[x/50 for x in range(1,2000,10)]
    V_vector = (1/2)*np.logspace(-2, 3, 2000) #[x/20 for x in range(1,500)]
    w_vector = np.logspace(-2, 6, 2000) #[exp(x/15) for x in range(1,200)]
    times = 2000
    answ = []
    counter = 0
    errorlog = []   
    while counter < times:   
        try: 
            answ += [make_database_core(random.choice(R_1),
        random.choice(R_2),
        random.choice(R_C),
        random.choice(R_E),
        random.choice(R_L),
            random.choice(C_1),
        random.choice(C_2),
        random.choice(V_vector),
        random.choice(w_vector),
                      )]
        except Exception as ins:
            errorlog.append(ins.args)
        counter += 1
    data = pd.DataFrame(answ,columns=['R1','R2','RC','RE','RL','C1','C2','V','W','D'])
    with open('database/database.csv', 'a') as f:
      data.to_csv(f, header=f.tell()==0, index=False)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='1':
        simulate_amplifier()
    elif sys.argv[1]=='2':
        simulate_amplification_factor_iterator()
    elif sys.argv[1]=='3':
        make_database()
