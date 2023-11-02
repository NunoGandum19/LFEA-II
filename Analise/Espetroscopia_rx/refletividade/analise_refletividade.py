import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from scipy import integrate

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

###################### Reading data #######################################################################################
# para a reflectividade relativa
NaCl_spectrum = reader("NaCl-spectrum.xry", 2.5, 60, 0.2, 1)
Si_spectrum = reader("Si_spectrum.xry", 2.5, 60, 0.2, 1)
LiF_spectrum = reader("LiF_spectrum.xry", 5, 40, 0.1, 1)
NaCl_X_spectrum = reader("NaCl_spectrum_x.xry", 2.5, 45, 0.1, 1)
NaCl_O_spectrum = reader("NaCl_spectrum_O.xry", 2.5, 45, 0.1, 4)
Al_spectrum_mau = reader("Al_Spectrum_mau.xry", 2.5, 45, 0.1, 1)
HOPG_spectrum = reader("HOPG_Spectrum.xry", 2.5, 45, 0.1, 1)
Safira_spectrum = reader("Safira_spectrum.xry", 10, 45, 0.1, 1)

spectrums = [NaCl_spectrum, Si_spectrum, LiF_spectrum, NaCl_X_spectrum, NaCl_O_spectrum, Al_spectrum_mau, HOPG_spectrum, Safira_spectrum]

# para a reflectividade absoluta
Fundo = reader("Nada_0.01mA.xry", -2.5, 2.5, 0.1, 5) # nosso
Fundo_deles = reader("Mari_nada0.01mA.xry", -2, 2, 0.1, 5) # deles
NaCl_reduced_deles = reader("Mari_NaCl_0.01mA.xry", 2.9, 30, 0.1, 3) # deles

###################### Plotting data #######################################################################################
for spec in spectrums:
    # plotter(spec)

######### Calculate Relative Reflectivity ##########################################################################
####################################################################################################################
# Obter Counts de cada pico

    Counts = []
for i in range(len(spectrums)):
    if i < 2:
        Counts.append(np.sum(2 * spectrums[i][1:,1]))
    else:
        Counts.append(np.sum(spectrums[i][1:,1]))

# Obter a refletividade relativa
refletividades_relativas = []
for i in range(len(Counts)):
    refletividades_relativas.append(Counts[i]/Counts[0])

print("Refletividade Relativa NaCl: ", refletividades_relativas[0] * 100, "%")
print("Refletividade Relativa Si: ", refletividades_relativas[1] * 100, "%")
print("Refletividade Relativa LiF: ", refletividades_relativas[2] * 100, "%")
print("Refletividade Relativa NaCl_X: ", refletividades_relativas[3] * 100, "%")
print("Refletividade Relativa NaCl_O: ", refletividades_relativas[4] * 100, "%")
print("Refletividade Relativa Al: ", refletividades_relativas[5] * 100, "%")
print("Refletividade Relativa HOPG: ", refletividades_relativas[6] * 100, "%")
print("Refletividade Relativa Safira: ", refletividades_relativas[7] * 100, "%")
print("\n")
    

"""#Selecionar valores so a partir de 10.1 (podemos mudar este valor)
new_specs = []
for spec in spectrums:
    for i in range(len(spec)):
        if spec[i][0] == 10.1:
            break
        new_spec = spec[i+1:]
    new_specs.append(new_spec)

new_new_specs = []
#Selecionar valores so até 39.9
for s in new_specs:
    for i in range(len(spec)):
        if s[i][0] == 39.9:
            break
        new_new_spec = s[:i+2]
    new_new_specs.append(new_new_spec)

#transformar esta merda em numpy arrays como deve ser 
x_piqi= np.array(new_new_specs[0])[:,0]
x_grande = np.array(new_new_specs[7])[:,0]
y_specs = []
for i in range(len(new_new_specs)): y_specs.append(np.array(new_new_specs[i])[:,1])

results = []
for spec in y_specs:
    if len(spec) == len(x_piqi):
        results.append(integrate.simps(spec, x_piqi))
    elif len(spec) == len(x_grande):
        results.append(integrate.simps(spec, x_grande))
    else:
        raise ValueError('O comprimento do array não é igual ao do x')

refletividades_relativas = []

for res in results: refletividades_relativas.append(res/results[0])

print('Refletividade Relativa Si: ', refletividades_relativas[1])
print('Refletividade Relativa LiF: ', refletividades_relativas[2])
print('Refletividade Relativa NaCl_X: ', refletividades_relativas[3])
print('Refletividade Relativa NaCl_O: ', refletividades_relativas[4])
print('Refletividade Relativa Al: ', refletividades_relativas[5])
print('Refletividade Relativa HOPG: ', refletividades_relativas[6])
print('Refletividade Relativa Safira: ', refletividades_relativas[7])"""


######################### Calculate the absolute reflectivity of NaCl ################################################################
######################################################################################################################################
plotter(Fundo, "Fundo")
#plotter(Fundo_deles, "Fundo deles")
plotter(NaCl_reduced_deles, "NaCl deles")

### Obter Counts 
"""Como o nº de counts nosso e deles é quase igual basta dividir para ter a refletividadade absoluta"""

counts_fundo = np.sum(Fundo[:,1])
#print('Counts Fundo: ', counts_fundo, "\n")
counts_NaCl_reduced = np.sum(NaCl_reduced_deles[1:,1])
#print('Counts NaCl: ', counts_NaCl_reduced, "\n")

##### Calculate the absolute reflectivity
refletividade_absoluta_NaCl = counts_NaCl_reduced/counts_fundo
print('Refletividade Absoluta NaCl: ', refletividade_absoluta_NaCl * 100 , '% \n')

############################ Calculate the absolute reflectivity of all crystals ###########################################################
############################################################################################################################################
Refletividades_absolutas = []
for i in range(len(refletividades_relativas)):
    Refletividades_absolutas.append(refletividades_relativas[i]*refletividade_absoluta_NaCl)

print("Refletividades Absoluta NaCl: ", Refletividades_absolutas[0])
print("Refletividades Absoluta Si: ", Refletividades_absolutas[1])
print("Refletividades Absoluta LiF: ", Refletividades_absolutas[2])
print("Refletividades Absoluta NaCl_X: ", Refletividades_absolutas[3])
print("Refletividades Absoluta NaCl_O: ", Refletividades_absolutas[4])
print("Refletividades Absoluta Al: ", Refletividades_absolutas[5])
print("Refletividades Absoluta HOPG: ", Refletividades_absolutas[6])
print("Refletividades Absoluta Safira: ", Refletividades_absolutas[7])

