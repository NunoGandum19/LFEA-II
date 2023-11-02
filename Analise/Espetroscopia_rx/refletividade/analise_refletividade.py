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

def erro_Counts(Nr_cts):
    return np.sqrt(Nr_cts)/Nr_cts

def erro_ref_rel(Nr_cts, Nr_cts_ref):
    return erro_Counts(Nr_cts)/Nr_cts_ref + erro_Counts(Nr_cts_ref)*(Nr_cts/Nr_cts_ref**2)

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
NaCl_O_spectrum = reader("NaCl_spectrum_O.xry", 2.5, 45, 0.1, 1)
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
erros_refletividades_relativas = []
for i in range(len(Counts)):
    refletividades_relativas.append(Counts[i]/Counts[0])
    erros_refletividades_relativas.append(erro_ref_rel(Counts[i], Counts[0]))

print("Refletividade Relativa NaCl: (", refletividades_relativas[0] * 100, "+-", erros_refletividades_relativas[0] * 100,") %")
print("Refletividade Relativa Si: ", refletividades_relativas[1] * 100, "+-", erros_refletividades_relativas[1] * 100,") %")
print("Refletividade Relativa LiF: ", refletividades_relativas[2] * 100, "+-", erros_refletividades_relativas[2] * 100,") %")
print("Refletividade Relativa NaCl_X: ", refletividades_relativas[3] * 100, "+-", erros_refletividades_relativas[3] * 100,") %")
print("Refletividade Relativa NaCl_O: ", refletividades_relativas[4] * 100, "+-", erros_refletividades_relativas[4] * 100,") %")
print("Refletividade Relativa Al: ", refletividades_relativas[5] * 100, "+-", erros_refletividades_relativas[5] * 100,") %")
print("Refletividade Relativa HOPG: ", refletividades_relativas[6] * 100, "+-", erros_refletividades_relativas[6] * 100,") %")
print("Refletividade Relativa Safira: ", refletividades_relativas[7] * 100, "+-", erros_refletividades_relativas[7] * 100,") %")
print("\n")

######################### Calculate the absolute reflectivity of NaCl ################################################################
######################################################################################################################################
#plotter(Fundo, "Fundo")
plotter(Fundo_deles, "Fundo deles")
plotter(NaCl_reduced_deles[1:], "NaCl deles")

counts_fundo_deles = np.sum(Fundo_deles[:,1])
counts_NaCl_reduced = np.sum(NaCl_reduced_deles[1:,1])

##### Calculate the absolute reflectivity
refletividade_absoluta_NaCl = counts_NaCl_reduced/counts_fundo_deles
erro_refletividade_absoluta_NaCl = erro_ref_rel(counts_fundo_deles, counts_NaCl_reduced)
print('Refletividade Absoluta NaCl: ', refletividade_absoluta_NaCl * 100 , '+-', erro_refletividade_absoluta_NaCl, '% \n')

############################ Calculate the absolute reflectivity of all crystals ###########################################################
############################################################################################################################################
Refletividades_absolutas = []
erros_refletividades_absolutas = []
for i in range(len(refletividades_relativas)):
    Refletividades_absolutas.append(refletividades_relativas[i]*refletividade_absoluta_NaCl)
    erros_refletividades_absolutas.append(erro_refletividade_absoluta_NaCl*refletividades_relativas[i] + erros_refletividades_relativas[i]*refletividade_absoluta_NaCl)

print("Refletividades Absoluta NaCl: ", Refletividades_absolutas[0] * 100, "+-", erros_refletividades_absolutas[0] * 100,") %" )
print("Refletividades Absoluta Si: ", Refletividades_absolutas[1] * 100, "+-", erros_refletividades_absolutas[1] * 100,") %" )
print("Refletividades Absoluta LiF: ", Refletividades_absolutas[2] * 100, "+-", erros_refletividades_absolutas[2] * 100,") %" )
print("Refletividades Absoluta NaCl_X: ", Refletividades_absolutas[3] * 100, "+-", erros_refletividades_absolutas[3] * 100,") %" )
print("Refletividades Absoluta NaCl_O: ", Refletividades_absolutas[4] * 100, "+-", erros_refletividades_absolutas[4] * 100,") %" )
print("Refletividades Absoluta Al: ", Refletividades_absolutas[5] * 100, "+-", erros_refletividades_absolutas[5] * 100,") %" )
print("Refletividades Absoluta HOPG: ", Refletividades_absolutas[6] * 100, "+-", erros_refletividades_absolutas[6] * 100,") %" )
print("Refletividades Absoluta Safira: ", Refletividades_absolutas[7] * 100, "+-", erros_refletividades_absolutas[7] * 100,") %" )

