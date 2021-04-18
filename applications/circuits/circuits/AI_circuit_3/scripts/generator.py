import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
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

def simulate_RC_filter():
    # Initialize the Circuit
    circuit = Circuit('RC low-pass filter')
    # Set the element's values
    V = 2
    w = 1E3
    R = 1
    C = 30
    fc = 1/2/3.14/(R*1000)/(C*1E-9)
    circuit.V('1', 1,circuit.gnd,f'DC 0 AC {V} SIN(0 {V} {w})')
    circuit.R('1',  1, 2, R@u_kΩ)
    circuit.C('1',  2, circuit.gnd, C@u_nF)
    # Simulate
    f = w/2/np.pi
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=((1/f)/10)@u_s, end_time=(4/f)@u_s)
    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(analysis['2'],label='out')
    ax.plot(analysis['1'],label='in')
    ax.legend()
    plt.savefig('gallery/RC-low-pass-filter_example.png')

def simulate_attenuation_factor_core(w,V,R,C):
    circuit = Circuit('RC low-pass filter')
    fc = 1/2/3.14/(R*1000)/(C*1E-9) 
    circuit.V('1', 1,circuit.gnd,f'DC 0 AC {V} SIN(0 {V} {w})')
    circuit.R('1',  1, 2, R@u_kΩ)
    circuit.C('1',  2, circuit.gnd, C@u_nF)
    f = w/2/np.pi
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=((1/f)/10)@u_s, end_time=(4/f)@u_s)
    if True:
        outputter = max(analysis['2'])
        del(circuit)
        return outputter
    return

def simulate_attenuation_factor_iterator():
    fig, ax = plt.subplots(figsize=(20,6))
    for (R,V) in list(itertools.product([1,2,3], [3])):
        for C in [10,30]:
            f_zero = 1/2/3.14/(R*1000)/(C*1E-9)
            try: 
                x = [i for i in range(1,100000,500)]
            except: break
            y = []
            for j in x:
                y.append(float(simulate_attenuation_factor_core(j,V,R,C)))
            ax.scatter(x,y,label=f'f={round(f_zero,-3)}')
            ax.plot(x,y)
            col = plt.gca().lines[-1].get_color()
            ax.axvline(linewidth=1.5,color=col, x=f_zero)
    ax.set_xlabel('freq (Hz)')
    ax.set_ylabel('Vpeak (V)')
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim([50,100000])
    plt.savefig('gallery/RC-low-pass-filter_attenuation-factor.png')

def make_database_core(w,V,R,C):
    circuit = Circuit('RC low-pass filter')
    # Include non-passive elements: 
    # D1N4148 is a (regular&cheap) Diode 
    circuit.include(spice_library['D1N4148']) 
    fc = 1/2/3.14/(R*1000)/(C*1E-9) # Cut Frequency!
    circuit.V('1', 1,circuit.gnd,f'DC 0 AC {V} SIN(0 {V} {w})')
    circuit.R('1',  1, 2, R@u_kΩ)
    circuit.C('1',  2, circuit.gnd, C@u_nF)
    circuit.X('diodus','D1N4148', 2, circuit.gnd)
    f = w
    treshold = 4
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=((1/f)/10)@u_s, end_time=(4/f)@u_s)
    outputter = int(float(max(-analysis['2']))>treshold)
    del(circuit)
    return [w,V,R,C,outputter] 
    
def make_database():
    W_vector = 2*np.logspace(2,5,100)
    V_vector = np.logspace(-2,2,100)
    R_vector = np.logspace(-2,3,100)
    C_vector = np.logspace(-2,2,100)
    answ = []
    times = 8000
    counter = 0
    errorlog = []
    while counter < times:
        try: 
            answ += [make_database_core(random.choice(W_vector),
        random.choice(V_vector),
        random.choice(R_vector),
        random.choice(C_vector),
                      )]
        except Exception as ins:
            errorlog.append(ins.args)
        counter += 1
    import pandas as pd

    data = pd.DataFrame(answ,columns=['W','V','R','C','D'])
    data.to_csv('database/database.csv', index=False)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='1':
        simulate_RC_filter()
    elif sys.argv[1]=='2':
        simulate_attenuation_factor_iterator()
    elif sys.argv[1]=='3':
        make_database()
