import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

################# FUNCTIONS ##################
def reader(name, beta_min, beta_max, delta_ang, delta_t):
    init_num = 17
    count_num = int((beta_max-beta_min)/delta_ang) + 17

    with open(name, "r", encoding='utf-8', errors='ignore') as file:
        counts = file.readlines()
    
    counts = counts[init_num:count_num]
    counts[0] = counts[0].split()[1]
    
    points = []
    #ct, ang = [], []
    for ii in range(len(counts)):
        angl = beta_min + ii*delta_ang
        pp = [float("{0:.1f}".format(angl)), float(counts[ii])]
        #ct.append(float(counts[ii]))
        #ang.append(float("{0:.1f}".format(angl)))
        points.append(pp)
    return np.array(points)

def plotter(array, Title = "Gráfico", clr = 'dodgerblue'):
    plt.plot(array[:,0], array[:,1], color = clr)
    plt.grid()
    plt.title(Title)
    plt.show()

def plotter_2(array1, array2, Title = "Gráfico"):
    plt.plot(array1[:,0], array1[:,1], label="Medição 1")
    plt.plot(array2[:,0], array2[:,1], label="Medição 2")
    plt.grid()
    plt.legend()
    plt.title(Title)
    plt.show()

def gaussian(x, amplitude, mean, stddev, cte = 0):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + cte

def double_gaussian(x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, cte = 0):
    return gaussian(x, amplitude1, mean1, stddev1) + gaussian(x, amplitude2, mean2, stddev2) + cte

def d_calculator(n, lambda_, teta):
    return n*lambda_/(2*np.sin(np.deg2rad(teta)))

def Potencia_Resolutiva(teta, delta_teta):
    return np.tan(np.deg2rad(teta))/delta_teta

######################################################################################
##################################### MAIN ###########################################
######################################################################################
### Constants
lambda_K_alpha = 71.080 # pm
lambda_K_beta = 63.095 # pm

### Reading data
"""
ler os ficheiros dos cristais com I = 0.1mA + NaCl com I = 0.1mA + do fundo
"""
reader('5c_Nada_Varr1.xry', -2.3, 2, 0.1, 2) # fundo
reader('5C_NACL_3Varr1.xry', -2.3, 30, 0.1, 2) # NaCl
reader('5c_Nacl2_2varr1.xry', -2.3, 30, 0.1, 2) # NaCl2 parte1
reader('5c_Nacl2_2varr2.xry', -2.3, 30, 0.1, 2) # NaCl2 parte2


### Plotting data
"""se quiseres"""

##### Calculate the absolute reflectivity of NaCl
"""
Dividir as counts de NaCl pelas do fundo
"""


##### Calculate the relative reflectivity of crystals
"""Dividir as counts dos cristais pelas do NaCl 1.0mA""" 

##### Calculate the absolute reflectivity of crystals
"""Multiplicar as reflectividades relativas dos cristais pela reflectividade absoluta do NaCl"""


