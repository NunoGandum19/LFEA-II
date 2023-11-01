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

"""
ler files para ref relativa 
"""
NaCl_spectrum = reader("dados_ref/NaCl_spectrum.xry", 2.5, 60, 0.2, 1)
Si_spectrum = reader("dados_ref/Si_spectrum.xry", 2.5, 60, 0.2, 1)
LiF_spectrum = reader("dados_ref/LiF_spectrum.xry", 5, 40, 0.1, 1)
NaCl_X_spectrum = reader("dados_ref/NaCl_spectrum_x.xry", 2.5, 45, 0.1, 1)
NaCl_O_spectrum = reader("dados_ref/NaCl_spectrum_O.xry", 2.5, 45, 0.1, 4)
Al_spectrum_mau = reader("dados_ref/Al_Spectrum_mau.xry", 2.5, 45, 0.1, 1)
HOPG_spectrum = reader("dados_ref/HOPG_Spectrum.xry", 2.5, 45, 0.1, 1)
Safira_spectrum = reader("dados_ref/Safira_spectrum.xry", 10, 45, 0.1, 1)
spectrums = [NaCl_spectrum, Si_spectrum, LiF_spectrum, NaCl_X_spectrum, NaCl_O_spectrum, Al_spectrum_mau, HOPG_spectrum, Safira_spectrum]
#Tem que começar todos em 10.1, vamos encontrar o 10.1 e depois cortar o array
#para começar em 10.1
for spectrum in spectrums:
    for i in range(len(spectrum)):
        if spectrum[i][0] == 10.1:
            break
    spectrum = spectrum[i:]


### Plotting data
"""se quiseres"""

##### Calculate the absolute reflectivity of NaCl
"""
Dividir as counts de NaCl pelas do fundo
"""
nacl_001 = reader("dados_ref/NaCl_0.01mA.xry", 5, 9, 0.1, 5)
nada_001 = reader("dados_ref/Nada_0.01mA.xry", -2.5, 2.5, 0.1, 5)

nacl_002 = reader("dados_ref/NaCl_0.02mA.xry", 5, 9, 0.1, 5)
nada_002 = reader("dados_ref/Nada_0.02mA.xry", -2.5, 2.5, 0.1, 5)

counts_nacl_001 = 0
counts_nacl_002 = 0
counts_nada_001 = 0
counts_nada_002 = 0

for i in range(1, len(nacl_001)):
    counts_nacl_001 += nacl_001[i][1]
    counts_nada_001 += nada_001[i][1]
    counts_nacl_002 += nacl_002[i][1]
    counts_nada_002 += nada_002[i][1]

reflec_abs_nacl_001 = counts_nacl_001/counts_nada_001
reflec_abs_nacl_002 = counts_nacl_002/counts_nada_002

print(reflec_abs_nacl_001)
print(reflec_abs_nacl_002)





##### Calculate the relative reflectivity of crystals
"""Dividir as counts dos cristais pelas do NaCl 1.0mA""" 

##### Calculate the absolute reflectivity of crystals
"""Multiplicar as reflectividades relativas dos cristais pela reflectividade absoluta do NaCl"""


