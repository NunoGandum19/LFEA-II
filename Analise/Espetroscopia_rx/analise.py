import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

def plotter(array, Title = "Gráfico"):
    plt.plot(array[:,0], array[:,1])
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

### Initializations
NaCl_K_alpha, NaCl_K_beta = [], []
NaCl_delta_K_alpha, NaCl_delta_K_beta = [], []

Si_K_alpha, Si_K_beta = [], []
Si_delta_K_alpha, Si_delta_K_beta = [], []

LiF_K_alpha, LiF_K_beta = [], []
LiF_delta_K_alpha, LiF_delta_K_beta = [], []

Al_K_alpha, Al_K_beta = [], []
Al_delta_K_alpha, Al_delta_K_beta = [], []

NaCl_X_K_alpha, NaCl_X_K_beta = [], []
NaCl_X_delta_K_alpha, NaCl_X_delta_K_beta = [], []

NaCl_O_K_alpha, NaCl_O_K_beta = [], []
NaCl_O_delta_K_alpha, NaCl_O_delta_K_beta = [], []

HOPG_K_alpha, HOPG_K_beta = [], []
HOPG_delta_K_alpha, HOPG_delta_K_beta = [], []

Safira_K_alpha, Safira_K_beta = [], []
Safira_delta_K_alpha, Safira_delta_K_beta = [], []

########################## NaCl (lab 3) #########################################################################################
##### Draw the graphs
NaCl_spectrum = reader("NaCl-spectrum.xry", 2.5, 60, 0.2, 1)
plotter(NaCl_spectrum, "NaCl Spectrum")

NaCl_1stGroup_1 = reader("NaCl_1stGroup_1.xry", 5, 9, 0.1, 6.5)
NaCl_1stGroup_2 = reader("NaCl_1stGroup_2.xry", 5, 9, 0.1, 6.5)
plotter_2(NaCl_1stGroup_1, NaCl_1stGroup_2, "NaCl 1st Group")

NaCl_2ndGroup_1 = reader("NaCl_2ndGroup_1.xry", 12, 16, 0.1, 6)
NaCl_2ndGroup_2 = reader("NaCl_2ndGroup_2.xry", 12, 16, 0.1, 6)
plotter_2(NaCl_2ndGroup_1, NaCl_2ndGroup_2, "NaCl 2nd Group")

NaCl_3rdGroup_1 = reader("NaCl_3rdGroup_1.xry", 18, 24, 0.1, 5)
NaCl_3rdGroup_2 = reader("NaCl_3rdGroup_2.xry", 18, 24, 0.1, 5)
plotter_2(NaCl_3rdGroup_1, NaCl_3rdGroup_2, "NaCl 3rd Group")

##### Obtain the K_alpha and K_beta peaks
Spectrums_NaCl = [NaCl_1stGroup_1, NaCl_1stGroup_2, NaCl_2ndGroup_1, NaCl_2ndGroup_2, NaCl_3rdGroup_1, NaCl_3rdGroup_2]
Spectrum_names_NaCl = ['NaCl_1stGroup_1', 'NaCl_1stGroup_2', 'NaCl_2ndGroup_1', 'NaCl_2ndGroup_2', 'NaCl_3rdGroup_1', 'NaCl_3rdGroup_2']
Init_Guess_NaCl = [[2000, 7.5, 0.5, 1000, 6.5, 0.5, 500], 
              [2000, 7.5, 0.5, 1000, 6.5, 0.5, 500], 
              [750, 14.75, 0.5, 300, 13.0, 0.5, 100], 
              [750, 14.75, 0.5, 300, 13.0, 0.5, 100], 
              [175, 22.5, 0.1, 75, 19.75, 0.5, 30], 
              [175, 22.5, 0.5, 75, 19.5, 0.5, 30]]
Bounds_NaCl = [([2000, 7.0, 0, 750, 6.0, 0, 400], [2500, 7.7, 1, 1250, 7.0, 1, 600]), 
          ([2000, 7.0, 0, 750, 6.0, 0, 400], [2500, 7.7, 1, 1250, 7.0, 1, 600]),
          ([700, 14.5, 0, 250, 12.5, 0, 50], [800, 15.0, 1, 350, 13.5, 1, 150]),
          ([700, 14.5, 0, 250, 12.5, 0, 50], [800, 15.0, 1, 350, 13.5, 1, 150]),
          ([150, 22.0, 0, 50, 19.5, 0, 10], [250, 23.0, 0.5, 100, 20.1, 1, 50]),
          ([150, 22.0, 0, 50, 19.0, 0, 10], [250, 23.0, 1, 100, 20.5, 1, 50])]
for i in range(len(Spectrums_NaCl)):
    p0 = Init_Guess_NaCl[i]  # Initial guess for the parameters
    bounds = Bounds_NaCl[i]  # Lower and upper bounds on parameters

    params, covariance = curve_fit(double_gaussian, Spectrums_NaCl[i][:, 0], Spectrums_NaCl[i][:, 1], p0=p0, bounds = bounds) # Fit the data
    amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, cte = params
    #print(f"Parameters for the fit of {Spectrum_names_NaCl[i]}: \n amplitude1 = {amplitude1} \n mean1 = {mean1} \n stddev1 = {stddev1} \n amplitude2 = {amplitude2} \n mean2 = {mean2} \n stddev2 = {stddev2} \n cte = {cte} \n")
    errors = np.sqrt(np.diag(covariance)) # Standard deviation errors on the parameters

    NaCl_K_alpha.append(mean1)
    NaCl_K_beta.append(mean2)
    NaCl_delta_K_alpha.append(stddev1)
    NaCl_delta_K_beta.append(stddev2)

    plt.figure(figsize=(8, 6))
    plt.plot(Spectrums_NaCl[i][:, 0], Spectrums_NaCl[i][:, 1], label='data', color = 'dodgerblue')
    plt.plot(Spectrums_NaCl[i][:, 0], double_gaussian(Spectrums_NaCl[i][:, 0], *params), label='fit', color = 'darkorange')
    plt.legend()
    plt.grid()
    plt.title(f"{Spectrum_names_NaCl[i]} fitted with a double gaussian")
    plt.xlabel('\u03B2 (°)')
    plt.ylabel('R(1/s)')
    plt.text(0.1, 0.9, f'Amplitude_\u03B1: {amplitude1:.2f} +- {"{:.5f}".format(errors[0])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.85, f'Mean_\u03B1: {mean1:.2f} +- {"{:.5f}".format(errors[1])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.8, f'Std_\u03B1: {stddev1:.2f} +- {"{:.5f}".format(errors[2])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.75, f'Amplitude_\u03B2: {amplitude2:.2f} +- {"{:.5f}".format(errors[3])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.7, f'Mean_\u03B2: {mean2:.2f} +- {"{:.5f}".format(errors[4])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.65, f'Std_\u03B2: {stddev2:.2f} +- {"{:.5f}".format(errors[5])}', transform=plt.gca().transAxes, color='black')
    plt.savefig(f"images/{Spectrum_names_NaCl[i]}_fitted.png")
    #plt.show()
    plt.close()

##### Calculate the Resolutive Power
NaCl_Potencia_Resolutiva_alpha = Potencia_Resolutiva(NaCl_K_alpha, NaCl_delta_K_alpha)
NaCl_Potencia_Resolutiva_beta = Potencia_Resolutiva(NaCl_K_beta, NaCl_delta_K_beta)

NaCl_Potencia_Resolutiva_alpha_mean, NaCl_Potencia_Resolutiva_beta_mean = [], []
for i in range(0, len(NaCl_Potencia_Resolutiva_alpha), 2):                          # tirar os valores médios para as duas aquisições
    NaCl_Potencia_Resolutiva_alpha_mean.append(np.mean(NaCl_Potencia_Resolutiva_alpha[i:i+2]))
    NaCl_Potencia_Resolutiva_beta_mean.append(np.mean(NaCl_Potencia_Resolutiva_beta[i:i+2]))

print(f"Resolutive Power for NaCl K_alpha: {NaCl_Potencia_Resolutiva_alpha_mean}")
print(f"Resolutive Power for NaCl K_beta: {NaCl_Potencia_Resolutiva_beta_mean}")

##### Calculate the d-spacing
n = np.array([1,1,2,2,3,3])
NaCl_d_alpha = d_calculator(n, lambda_K_alpha, np.array(NaCl_K_alpha))
NaCl_d_beta = d_calculator(n, lambda_K_beta, np.array(NaCl_K_beta))

NaCl_d_alpha_mean, NaCl_d_beta_mean = [], []
for i in range(0, len(NaCl_d_alpha), 2):
    NaCl_d_alpha_mean.append(np.mean(NaCl_d_alpha[i:i+2]))
    NaCl_d_beta_mean.append(np.mean(NaCl_d_beta[i:i+2]))

d_NaCl = np.mean(np.concatenate((NaCl_d_alpha_mean, NaCl_d_beta_mean)))
print(f"d-spacing for NaCl: {d_NaCl}\n")

########################## Si (lab 4) ####################################################################################
##### Draw the graphs
Si_spectrum = reader("Si_spectrum.xry", 2.5, 60, 0.2, 1)
plotter(Si_spectrum, "Si Spectrum")

Si_1stGroup_1 = reader("Si_1stGroup_1.xry", 15, 21, 0.1, 5)
Si_1stGroup_2 = reader("Si_1stGroup_2.xry", 15, 21, 0.1, 5)
plotter_2(Si_1stGroup_1, Si_1stGroup_2, "Si 1st Group")

Si_2ndGroup_1 = reader("Si_2ndGroup_1.xry", 23, 30, 0.1, 5)
Si_2ndGroup_2 = reader("Si_2ndGroup_2.xry", 23, 30, 0.1, 5)
plotter_2(Si_2ndGroup_1, Si_2ndGroup_2, "Si 2nd Group")

Si_3rdGroup_1 = reader("Si_3rdGroup_1.xry", 34, 40, 0.1, 3)  # ver bem estes gráficos
Si_3rdGroup_2 = reader("Si_3rdGroup_2.xry", 30, 37, 0.1, 4)
plotter_2(Si_3rdGroup_1, Si_3rdGroup_2, "Si 3rd Group\n")

### Obtain the K_alpha and K_beta peaks
Spectrums_Si = [Si_1stGroup_1, Si_1stGroup_2, Si_2ndGroup_1, Si_2ndGroup_2]
Spectrum_names_Si = ['Si_1stGroup_1', 'Si_1stGroup_2', 'Si_2ndGroup_1', 'Si_2ndGroup_2']
Init_Guess_Si = [[100, 19.5, 0.5, 60, 17, 0.5, 25], 
              [100, 19.5, 0.5, 60, 17, 0.5, 25], 
              [70, 27.5, 0.5, 40, 24, 0.5, 20], 
              [70, 27.5, 0.5, 40, 24, 0.5, 20]]
Bounds_Si = [([60, 19, 0, 20, 16.5, 0, 15], [120, 20, 1, 80, 17.5, 1, 40]),
            ([60, 19, 0, 20, 16.5, 0, 15], [120, 20, 1, 80, 17.5, 1, 40]),
            ([50, 27, 0, 15, 23.9, 0, 15], [80, 28, 1, 40, 24.25, 0.5, 25]),
            ([50, 27, 0, 15, 23.9, 0, 15], [80, 28, 1, 40, 24.25, 0.5, 25])]
for i in range(len(Spectrums_Si)):
    p0 = Init_Guess_Si[i]  # Initial guess for the parameters
    bounds = Bounds_Si[i]  # Lower and upper bounds on parameters

    params, covariance = curve_fit(double_gaussian, Spectrums_Si[i][1:, 0], Spectrums_Si[i][1:, 1], p0=p0, bounds = bounds) # Fit the data
    amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, cte = params
    #print(f"Parameters for the fit of {Spectrum_names_Si[i]}: \n amplitude1 = {amplitude1} \n mean1 = {mean1} \n stddev1 = {stddev1} \n amplitude2 = {amplitude2} \n mean2 = {mean2} \n stddev2 = {stddev2} \n cte = {cte} \n")
    errors = np.sqrt(np.diag(covariance)) # Standard deviation errors on the parameters
    
    Si_K_alpha.append(mean1)
    Si_K_beta.append(mean2)
    Si_delta_K_alpha.append(stddev1)
    Si_delta_K_beta.append(stddev2)

    plt.figure(figsize=(8, 6))
    plt.plot(Spectrums_Si[i][:, 0], Spectrums_Si[i][:, 1], label='data', color = 'limegreen')
    plt.plot(Spectrums_Si[i][:, 0], double_gaussian(Spectrums_Si[i][:, 0], *params), label='fit', color = 'darkorange')
    plt.legend()
    plt.grid()
    plt.title(f"{Spectrum_names_Si[i]} fitted with a double gaussian")
    plt.xlabel('\u03B2 (°)')
    plt.ylabel('R(1/s)')
    plt.text(0.1, 0.9, f'Amplitude_\u03B1: {amplitude1:.2f} +- {"{:.5f}".format(errors[0])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.85, f'Mean_\u03B1: {mean1:.2f} +- {"{:.5f}".format(errors[1])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.8, f'Std_\u03B1: {stddev1:.2f} +- {"{:.5f}".format(errors[2])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.75, f'Amplitude_\u03B2: {amplitude2:.2f} +- {"{:.5f}".format(errors[3])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.7, f'Mean_\u03B2: {mean2:.2f} +- {"{:.5f}".format(errors[4])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.65, f'Std_\u03B2: {stddev2:.2f} +- {"{:.5f}".format(errors[5])}', transform=plt.gca().transAxes, color='black')
    plt.savefig(f"images/{Spectrum_names_Si[i]}_fitted.png")
    #plt.show()
    plt.close()

### Calculate the Resolutive Power
Si_Potencia_Resolutiva_alpha = Potencia_Resolutiva(Si_K_alpha, Si_delta_K_alpha)
Si_Potencia_Resolutiva_beta = Potencia_Resolutiva(Si_K_beta, Si_delta_K_beta)

Si_Potencia_Resolutiva_alpha_mean, Si_Potencia_Resolutiva_beta_mean = [], []
for i in range(0, len(Si_Potencia_Resolutiva_alpha), 2):                          # tirar os valores médios para as duas aquisições
    Si_Potencia_Resolutiva_alpha_mean.append(np.mean(Si_Potencia_Resolutiva_alpha[i:i+2]))
    Si_Potencia_Resolutiva_beta_mean.append(np.mean(Si_Potencia_Resolutiva_beta[i:i+2]))

print(f"Resolutive Power for Si K_alpha: {Si_Potencia_Resolutiva_alpha_mean}")
print(f"Resolutive Power for Si K_beta: {Si_Potencia_Resolutiva_beta_mean}")

### Calculate the d-spacing
n = np.array([1,1,2,2])
Si_d_alpha = d_calculator(n, lambda_K_alpha, np.array(Si_K_alpha))
Si_d_beta = d_calculator(n, lambda_K_beta, np.array(Si_K_beta))

Si_d_alpha_mean, Si_d_beta_mean = [], []
for i in range(0, len(Si_d_alpha), 2):
    Si_d_alpha_mean.append(np.mean(Si_d_alpha[i:i+2]))
    Si_d_beta_mean.append(np.mean(Si_d_beta[i:i+2]))

d_Si = np.mean(np.concatenate((Si_d_alpha_mean, Si_d_beta_mean)))
print(f"d-spacing for Si: {d_Si}\n")

################################### LiF (lab 4) #######################################################################
# Draw the graphs
LiF_spectrum = reader("LiF_spectrum.xry", 5, 40, 0.1, 1)
plotter(LiF_spectrum, "LiF Spectrum")

LiF_1stGroup = reader("LiF_1stGroup.xry", 6, 12, 0.1, 5)
plotter(LiF_1stGroup, "LiF 1st Group")

LiF_2ndGroup = reader("LiF_2ndGroup.xry", 16.5, 21.5, 0.1, 6)
plotter(LiF_2ndGroup, "LiF 2nd Group")

LiF_3rdGroup = reader("LiF_3rdGroup.xry", 26, 32.5, 0.1, 5)
plotter(LiF_3rdGroup, "LiF 3rd Group")

# Obtain the K_alpha and K_beta peaks
Spectrums_LiF = [LiF_1stGroup, LiF_2ndGroup, LiF_3rdGroup]
Spectrum_names_LiF = ['LiF_1stGroup', 'LiF_2ndGroup', 'LiF_3rdGroup']
Init_Guess_LiF = [[66, 10, 0.5, 0, 7.5, 0, 75], 
                  [60, 20, 0.5, 20, 17.75, 0.5, 70], 
                  [60, 31.5, 0.5, 15, 27.5, 0.5, 40]]
Bounds_LiF = [([50, 9.5, 0, 0, 7, 0, 40], [100, 10.5, 1, 20, 8.5, 3, 90]),
              ([60, 19.5, 0, 10, 17, 0, 30], [100, 20.5, 1, 45, 18.1, 1, 80]),
              ([50, 31, 0, 10, 27, 0, 20], [100, 32, 1, 45, 28, 1, 60])]
for i in range(len(Spectrums_LiF)):
    p0 = Init_Guess_LiF[i]  # Initial guess for the parameters
    bounds = Bounds_LiF[i]  # Lower and upper bounds on parameters

    params, covariance = curve_fit(double_gaussian, Spectrums_LiF[i][:, 0], Spectrums_LiF[i][:, 1], p0=p0, bounds = bounds) # Fit the data
    amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, cte = params
    #print(f"Parameters for the fit of {Spectrum_names_LiF[i]}: \n amplitude1 = {amplitude1} \n mean1 = {mean1} \n stddev1 = {stddev1} \n amplitude2 = {amplitude2} \n mean2 = {mean2} \n stddev2 = {stddev2} \n cte = {cte} \n")
    errors = np.sqrt(np.diag(covariance)) # Standard deviation errors on the parameters

    LiF_K_alpha.append(mean1)
    LiF_K_beta.append(mean2)
    LiF_delta_K_alpha.append(stddev1)
    LiF_delta_K_beta.append(stddev2)

    plt.figure(figsize=(8, 6))
    plt.plot(Spectrums_LiF[i][:, 0], Spectrums_LiF[i][:, 1], label='data', color = 'indigo')
    plt.plot(Spectrums_LiF[i][:, 0], double_gaussian(Spectrums_LiF[i][:, 0], *params), label='fit', color = 'darkorange')
    plt.legend()
    plt.grid()
    plt.title(f"{Spectrum_names_LiF[i]} fitted with a double gaussian")
    plt.xlabel('\u03B2 (°)')
    plt.ylabel('R(1/s)')
    plt.text(0.1, 0.9, f'Amplitude_\u03B1: {amplitude1:.2f} +- {"{:.5f}".format(errors[0])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.85, f'Mean_\u03B1: {mean1:.2f} +- {"{:.5f}".format(errors[1])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.8, f'Std_\u03B1: {stddev1:.2f} +- {"{:.5f}".format(errors[2])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.75, f'Amplitude_\u03B2: {amplitude2:.2f}  +- {"{:.5f}".format(errors[3])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.7, f'Mean_\u03B2: {mean2:.2f} +- {"{:.5f}".format(errors[4])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.65, f'Std_\u03B2: {stddev2:.2f} +- {"{:.5f}".format(errors[5])}', transform=plt.gca().transAxes, color='black')
    plt.savefig(f"images/{Spectrum_names_LiF[i]}_fitted.png")
    #plt.show()
    plt.close()

##### Calculate the Resolutive Power
LiF_Potencia_Resolutiva_alpha = Potencia_Resolutiva(LiF_K_alpha, LiF_delta_K_alpha)
LiF_Potencia_Resolutiva_beta = Potencia_Resolutiva(LiF_K_beta[1:], LiF_delta_K_beta[1:])

print(f"Resolutive Power for LiF K_alpha: {LiF_Potencia_Resolutiva_alpha}")
print(f"Resolutive Power for LiF K_beta: {LiF_Potencia_Resolutiva_beta}")

##### Calculate the d-spacing
n = np.array([1,2,3])
LiF_d_alpha = d_calculator(n, lambda_K_alpha, np.array(LiF_K_alpha))
LiF_d_beta = d_calculator(n[1:], lambda_K_beta, np.array(LiF_K_beta[1:]))

d_LiF = np.mean(np.concatenate((LiF_d_alpha, LiF_d_beta)))
print(f"d-spacing for LiF: {d_LiF}\n")

################## Al (lab 5) ###################################################################################
##### Draw the graphs
Al_spectrum_mau = reader("Al_Spectrum_mau.xry", 2.5, 45, 0.1, 1)
plotter(Al_spectrum_mau, "Al Spectrum")

Al_1stGroup_mau_1 = reader("al_1st_group_1_mau.xry", 4, 11, 0.1, 5)
Al_1stGroup_mau_2 = reader("al_1st_group_2_mau.xry", 4, 11, 0.1, 5)
plotter_2(Al_1stGroup_mau_1, Al_1stGroup_mau_2, "Al 1st Group")

##### Obtain the K_alpha and K_beta peaks
spectrum_Al = [Al_1stGroup_mau_1, Al_1stGroup_mau_2]
spectrum_names_Al = ['Al_1stGroup_1', 'Al_1stGroup_2']
Init_Guess_Al = [[120, 9.5, 0.5, 50, 8.5, 0.1, 110],
                 [120, 9.5, 0.5, 50, 8.5, 0.1, 110]]
Bounds_Al = [([100, 9, 0, 0, 8.35, 0, 50], [200, 10, 1, 100, 9, 0.5, 120]),
             ([100, 9, 0, 0, 8.35, 0, 50], [200, 10, 1, 100, 9, 0.5, 120])]
for i in range(len(spectrum_Al)):
    p0  = Init_Guess_Al[i]  # Initial guess for the parameters
    bounds = Bounds_Al[i]  # Lower and upper bounds on parameters

    params, covariance = curve_fit(double_gaussian, spectrum_Al[i][:, 0], spectrum_Al[i][:, 1], p0=p0, bounds = bounds) # Fit the data
    amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, cte = params
    errors = np.sqrt(np.diag(covariance)) # Standard deviation errors on the parameters

    Al_K_alpha.append(mean1)
    Al_K_beta.append(mean2)
    Al_delta_K_alpha.append(stddev1)
    Al_delta_K_beta.append(stddev2)

    plt.figure(figsize=(8, 6))
    plt.plot(spectrum_Al[i][:, 0], spectrum_Al[i][:, 1], label='data', color = 'darkolivegreen')
    plt.plot(spectrum_Al[i][:, 0], double_gaussian(spectrum_Al[i][:, 0], *params), label='fit', color = 'darkorange')
    plt.legend()
    plt.grid()
    plt.title(f"{spectrum_names_Al[i]} fitted with a double gaussian")
    plt.xlabel('\u03B2 (°)')
    plt.ylabel('R(1/s)')
    plt.text(0.1, 0.9, f'Amplitude_\u03B1: {amplitude1:.2f} +- {"{:.5f}".format(errors[0])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.85, f'Mean_\u03B1: {mean1:.2f} +- {"{:.5f}".format(errors[1])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.8, f'Std_\u03B1: {stddev1:.2f} +- {"{:.5f}".format(errors[2])}', transform=plt.gca().transAxes, color='black')  
    plt.text(0.1, 0.75, f'Amplitude_\u03B2: {amplitude2:.2f} +- {"{:.5f}".format(errors[3])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.7, f'Mean_\u03B2: {mean2:.2f} +- {"{:.5f}".format(errors[4])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.65, f'Std_\u03B2: {stddev2:.2f} +- {"{:.5f}".format(errors[5])}', transform=plt.gca().transAxes, color='black')
    plt.savefig(f"images/{spectrum_names_Al[i]}_fitted.png")
    #plt.show()
    plt.close()

##### Calculate the Resolutive Power
Al_Potencia_Resolutiva_alpha = Potencia_Resolutiva(Al_K_alpha, Al_delta_K_alpha)
Al_Potencia_Resolutiva_beta = Potencia_Resolutiva(Al_K_beta, Al_delta_K_beta)

Al_Potencia_Resolutiva_alpha_mean = np.mean(Al_Potencia_Resolutiva_alpha)
Al_Potencia_Resolutiva_beta_mean = np.mean(Al_Potencia_Resolutiva_beta)

print(f"Resolutive Power for Al K_alpha: {Al_Potencia_Resolutiva_alpha_mean}")
print(f"Resolutive Power for Al K_beta: {Al_Potencia_Resolutiva_beta_mean}")

##### Calculate the d-spacing
n = np.array([1,1])
Al_d_alpha = d_calculator(n, lambda_K_alpha, np.array(Al_K_alpha))
Al_d_beta = d_calculator(n, lambda_K_beta, np.array(Al_K_beta))

d_Al = np.mean(np.concatenate((Al_d_alpha, Al_d_beta)))
print(f"d-spacing for Al: {d_Al}\n")

################## NaCl_X (lab 5) #############################################################################
##### Draw the graphs
NaCl_X_spectrum = reader("NaCl_spectrum_x.xry", 2.5, 45, 0.1, 1)
plotter(NaCl_X_spectrum, "NaCl_X Spectrum")

NaCl_X_1stGroup_1 = reader("NaCl_x_1stGroup_1.xry", 9, 16, 0.1, 5)
NaCl_X_1stGroup_2 = reader("NaCl_x_1stGroup_2.xry", 9, 16, 0.1, 5)
plotter_2(NaCl_X_1stGroup_1, NaCl_X_1stGroup_2, "NaCl_X 1st Group")

##### Obtain the K_alpha and K_beta peaks
Spectrums_NaCl_X = [NaCl_X_1stGroup_1, NaCl_X_1stGroup_2]
Spectrum_names_NaCl_X = ['NaCl_X_1stGroup_1', 'NaCl_X_1stGroup_2']
Init_Guess_NaCl_X = [[25, 13.5, 0.5, 10, 12, 0.5, 15], 
                     [25, 13.5, 0.5, 10, 12, 0.5, 15]]
Bounds_NaCl_X = [([15, 13, 0, 0, 11.5, 0, 10], [30, 14, 1, 20, 12.5, 1, 25]),
                 ([15, 13, 0, 0, 11.5, 0, 10], [30, 14, 1, 20, 12.5, 1, 25])]
for i in range(len(Spectrums_NaCl_X)):
    p0 = Init_Guess_NaCl_X[i]  # Initial guess for the parameters
    bounds = Bounds_NaCl_X[i]  # Lower and upper bounds on parameters

    params, covariance = curve_fit(double_gaussian, Spectrums_NaCl_X[i][:, 0], Spectrums_NaCl_X[i][:, 1], p0=p0, bounds = bounds) # Fit the data
    amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, cte = params
    #print(f"Parameters for the fit of {Spectrum_names_NaCl_X[i]}: \n amplitude1 = {amplitude1} \n mean1 = {mean1} \n stddev1 = {stddev1} \n amplitude2 = {amplitude2} \n mean2 = {mean2} \n stddev2 = {stddev2} \n cte = {cte} \n")
    errors = np.sqrt(np.diag(covariance)) # Standard deviation errors on the parameters
    
    NaCl_X_K_alpha.append(mean1)
    NaCl_X_K_beta.append(mean2)
    NaCl_X_delta_K_alpha.append(stddev1)
    NaCl_X_delta_K_beta.append(stddev2)

    plt.figure(figsize=(8, 6))
    plt.plot(Spectrums_NaCl_X[i][:, 0], Spectrums_NaCl_X[i][:, 1], label='data', color = 'darkturquoise')
    plt.plot(Spectrums_NaCl_X[i][:, 0], double_gaussian(Spectrums_NaCl_X[i][:, 0], *params), label='fit', color = 'darkorange')
    plt.legend()
    plt.grid()
    plt.title(f"{Spectrum_names_NaCl_X[i]} fitted with a double gaussian")
    plt.xlabel('\u03B2 (°)')
    plt.ylabel('R(1/s)')
    plt.text(0.1, 0.9, f'Amplitude_\u03B1: {amplitude1:.2f}  +- {"{:.5f}".format(errors[0])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.85, f'Mean_\u03B1: {mean1:.2f} +- {"{:.5f}".format(errors[1])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.8, f'Std_\u03B1: {stddev1:.2f} +- {"{:.5f}".format(errors[2])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.75, f'Amplitude_\u03B2: {amplitude2:.2f} +- {"{:.5f}".format(errors[3])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.7, f'Mean_\u03B2: {mean2:.2f} +- {"{:.5f}".format(errors[4])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.65, f'Std_\u03B2: {stddev2:.2f} +- {"{:.5f}".format(errors[5])}', transform=plt.gca().transAxes, color='black')
    plt.savefig(f"images/{Spectrum_names_NaCl_X[i]}_fitted.png")
    #plt.show()
    plt.close()

##### Calculate the Resolutive Power
NaCl_X_Potencia_Resolutiva_alpha = Potencia_Resolutiva(NaCl_X_K_alpha, NaCl_X_delta_K_alpha)
NaCl_X_Potencia_Resolutiva_beta = Potencia_Resolutiva(NaCl_X_K_beta, NaCl_X_delta_K_beta)

NaCl_X_Potencia_Resolutiva_alpha_mean = np.mean(NaCl_X_Potencia_Resolutiva_alpha)
NaCl_X_Potencia_Resolutiva_beta_mean = np.mean(NaCl_X_Potencia_Resolutiva_beta)

print(f"Resolutive Power for NaCl_X K_alpha: {NaCl_X_Potencia_Resolutiva_alpha_mean}")
print(f"Resolutive Power for NaCl_X K_beta: {NaCl_X_Potencia_Resolutiva_beta_mean}")

##### Calculate the d-spacing
n=np.array([2, 2])  # assumimos que é do 2º grupo porque do 1º grupo estava muito mau
NaCl_X_d_alpha = d_calculator(n, lambda_K_alpha, np.array(NaCl_X_K_alpha))
NaCl_X_d_beta = d_calculator(n, lambda_K_beta, np.array(NaCl_X_K_beta))

NaCl_X_d_alpha_mean = np.mean(NaCl_X_d_alpha)
NaCl_X_d_beta_mean = np.mean(NaCl_X_d_beta)

d_NaCl_X = np.mean(np.concatenate((NaCl_X_d_alpha, NaCl_X_d_beta)))
print(f"d-spacing for NaCl_X: {d_NaCl_X} \n")

################## NaCl_O (lab 5) ####################################################################################
##### Draw the graphs
NaCl_O_spectrum = reader("NaCl_spectrum_O.xry", 2.5, 45, 0.1, 4)
plotter(NaCl_O_spectrum, "NaCl_O Spectrum")

NaCl_O_1stGroup_1 = reader("NaCl_O_1stGroup_1.xry", 2.5, 7.5, 0.1, 5)
NaCl_O_1stGroup_2 = reader("NaCl_O_1stGroup_2.xry", 2.5, 7.5, 0.1, 5)
plotter_2(NaCl_O_1stGroup_1, NaCl_O_1stGroup_2, "NaCl_O 1st Group")

NaCl_O_2ndGroup_1 = reader("NaCl_O_2ndGroup_1.xry", 7.5, 13, 0.1, 5)
NaCl_O_2ndGroup_2 = reader("NaCl_O_2ndGroup_2.xry", 7.5, 13, 0.1, 5)
plotter_2(NaCl_O_2ndGroup_1, NaCl_O_2ndGroup_2, "NaCl_O 2nd Group")

NaCl_O_3rdGroup_1 = reader("NaCl_O_3rdGroup_1.xry", 13, 19, 0.1, 5)
NaCl_O_3rdGroup_2 = reader("NaCl_O_3rdGroup_2.xry", 13, 19, 0.1, 5)
plotter_2(NaCl_O_3rdGroup_1, NaCl_O_3rdGroup_2, "NaCl_O 3rd Group")

##### Obtain the K_alpha and K_beta peaks
Spectrums_NaCl_O = [NaCl_O_1stGroup_1, NaCl_O_1stGroup_2, NaCl_O_2ndGroup_1, NaCl_O_2ndGroup_2, NaCl_O_3rdGroup_1, NaCl_O_3rdGroup_2]
Spectrum_names_NaCl_O = ['NaCl_O_1stGroup_1', 'NaCl_O_1stGroup_2', 'NaCl_O_2ndGroup_1', 'NaCl_O_2ndGroup_2', 'NaCl_O_3rdGroup_1', 'NaCl_O_3rdGroup_2']
Init_Guess_NaCl_O = [[30, 6, 0.5, 0, 0, 0, 30],
                     [30, 6, 0.5, 0, 0, 0, 30],
                     [80, 11, 0.2, 40, 10, 0.2, 30],
                     [80, 11, 0.2, 40, 10, 0.2, 30],
                     [50, 16.5, 0.5, 20, 14.5, 0.1, 35],
                     [50, 16.5, 0.5, 20, 14.5, 0.1, 35]]
Bounds_NaCl_O = [([20, 5.5, 0, 0, 0, 0, 20], [40, 6.5, 1, 0.1, 0.1, 0.1, 40]),
                    ([20, 5.5, 0, 0, 0, 0, 20], [40, 6.5, 1, 0.1, 0.1, 0.1, 40]),
                    ([75, 10.5, 0, 30, 9.5, 0, 20], [100, 11.5, 0.5, 50, 10.5, 0.5, 45]),
                    ([75, 10.5, 0, 30, 9.5, 0, 20], [100, 11.5, 0.5, 50, 10.5, 0.5, 45]),
                    ([40, 16, 0, 10, 14.3, 0, 20], [65, 17, 1, 45, 14.6, 1, 50]),
                    ([40, 16, 0, 10, 14.3, 0, 20], [65, 17, 1, 45, 14.6, 1, 50])]
for i in range(len(Spectrums_NaCl_O)):
    p0 = Init_Guess_NaCl_O[i]  # Initial guess for the parameters
    bounds = Bounds_NaCl_O[i]  # Lower and upper bounds on parameters

    params, covariance = curve_fit(double_gaussian, Spectrums_NaCl_O[i][15:, 0], Spectrums_NaCl_O[i][15:, 1], p0=p0, bounds = bounds) # Fit the data
    amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, cte = params
    #print(f"Parameters for the fit of {Spectrum_names_NaCl_O[i]}: \n amplitude1 = {amplitude1} \n mean1 = {mean1} \n stddev1 = {stddev1} \n amplitude2 = {amplitude2} \n mean2 = {mean2} \n stddev2 = {stddev2} \n cte = {cte} \n")
    errors = np.sqrt(np.diag(covariance)) # Standard deviation errors on the parameters
    
    NaCl_O_K_alpha.append(mean1)
    NaCl_O_K_beta.append(mean2)
    NaCl_O_delta_K_alpha.append(stddev1)
    NaCl_O_delta_K_beta.append(stddev2)

    plt.figure(figsize=(8, 6))
    plt.plot(Spectrums_NaCl_O[i][:, 0], Spectrums_NaCl_O[i][:, 1], label='data', color = 'darkgreen')
    plt.plot(Spectrums_NaCl_O[i][:, 0], double_gaussian(Spectrums_NaCl_O[i][:, 0], *params), label='fit', color = 'darkorange')
    plt.legend()
    plt.grid()
    plt.title(f"{Spectrum_names_NaCl_O[i]} fitted with a double gaussian")
    plt.xlabel('\u03B2 (°)')
    plt.ylabel('R(1/s)')
    plt.text(0.1, 0.9, f'Amplitude_\u03B1: {amplitude1:.2f} +- {"{:.5f}".format(errors[0])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.85, f'Mean_\u03B1: {mean1:.2f} +- {"{:.5f}".format(errors[1])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.8, f'Std_\u03B1: {stddev1:.2f} +- {"{:.5f}".format(errors[2])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.75, f'Amplitude_\u03B2: {amplitude2:.2f} +- {"{:.5f}".format(errors[3])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.7, f'Mean_\u03B2: {mean2:.2f} +- {"{:.5f}".format(errors[4])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.65, f'Std_\u03B2: {stddev2:.2f} +- {"{:.5f}".format(errors[5])}', transform=plt.gca().transAxes, color='black')
    plt.savefig(f"images/{Spectrum_names_NaCl_O[i]}_fitted.png")
    #plt.show()
    plt.close()

##### Calculate the Resolutive Power
NaCl_O_Potencia_Resolutiva_alpha = Potencia_Resolutiva(NaCl_O_K_alpha[2:], NaCl_O_delta_K_alpha[2:])
NaCl_O_Potencia_Resolutiva_beta = Potencia_Resolutiva(NaCl_O_K_beta[2:], NaCl_O_delta_K_beta[2:])

NaCl_O_Potencia_Resolutiva_alpha_mean, NaCl_O_Potencia_Resolutiva_beta_mean = [], []
for i in range(0, len(NaCl_O_Potencia_Resolutiva_alpha), 2):
    NaCl_O_Potencia_Resolutiva_alpha_mean.append(np.mean(NaCl_O_Potencia_Resolutiva_alpha[i:i+2]))
    NaCl_O_Potencia_Resolutiva_beta_mean.append(np.mean(NaCl_O_Potencia_Resolutiva_beta[i:i+2]))

print(f"Resolutive Power for NaCl_O K_alpha: {NaCl_O_Potencia_Resolutiva_alpha_mean}")
print(f"Resolutive Power for NaCl_O K_beta: {NaCl_O_Potencia_Resolutiva_beta_mean}")

##### Calculate the d-spacing
n = np.array([2,2,3,3])

NaCl_O_d_alpha = d_calculator(n, lambda_K_alpha, np.array(NaCl_O_K_alpha[2:]))
NaCl_O_d_beta = d_calculator(n, lambda_K_beta, np.array(NaCl_O_K_beta)[2:])

NaCl_O_d_alpha_mean, NaCl_O_d_beta_mean = [], []
for i in range(0, len(NaCl_O_d_alpha), 2):
    NaCl_O_d_alpha_mean.append(np.mean(NaCl_O_d_alpha[i:i+2]))
    NaCl_O_d_beta_mean.append(np.mean(NaCl_O_d_beta[i:i+2]))

d_NaCl_O = np.mean(np.concatenate((NaCl_O_d_alpha_mean, NaCl_O_d_beta_mean)))
print(f"d-spacing for NaCl_O: {d_NaCl_O} \n")


###################### Refletividade absoluta do NaCl (lab 6) #####################################################

###################### Estudar o espetro de Si em função da direção ################################################
'''Si_rodado = reader("Si_spectrum_rodado.xry", 10, 40, 0.1, 1)
plotter(Si_rodado)
'''

################################# HOPG (lab 6) ###################################################################
##### Draw the graphs
HOPG_spectrum = reader("HOPG_Spectrum.xry", 2.5, 45, 0.1, 1)
plotter(HOPG_spectrum, "HOPG Spectrum")

HOPG_1stGroup = reader("HOPG_1st_Group.xry", 4.5, 8, 0.1, 5)
plotter(HOPG_1stGroup, "HOPG 1st Group")

HOPG_2ndGroup = reader("HOPG_2nd_Group.xry", 10, 14, 0.1, 5)
plotter(HOPG_2ndGroup, "HOPG 2nd Group")

HOPG_3rdGroup = reader("HOPG_3rd_Group.xry", 16.5, 20.5, 0.1, 5)
plotter(HOPG_3rdGroup, "HOPG 3rd Group")

##### Obtain the K_alpha and K_beta peaks
Spectrums_HOPG = [HOPG_1stGroup, HOPG_2ndGroup, HOPG_3rdGroup]
Spectrum_names_HOPG = ['HOPG_1stGroup', 'HOPG_2ndGroup', 'HOPG_3rdGroup']
Init_Guess_HOPG = [[400, 7, 0.5, 0, 0, 0, 300],
                   [600, 13, 0.5, 100, 11.7, 0.5, 200],
                   [150, 19.2, 0.5, 50, 17.25, 0.5, 100]]
Bounds_HOPG = [([300, 6.5, 0, 0, 0, 0, 250], [600, 7.5, 1, 0.1, 0.1, 0.1, 400]),
                ([500, 12.5, 0, 50, 11.5, 0, 150], [750, 13.5, 1, 150, 12, 1, 300]),
                ([100, 19, 0, 20, 17, 0, 50], [200, 20, 1, 80, 17.5, 1, 150])]
for i in range(len(Spectrums_HOPG)):
    p0 = Init_Guess_HOPG[i]  # Initial guess for the parameters
    bounds = Bounds_HOPG[i]  # Lower and upper bounds on parameters

    params, covariance = curve_fit(double_gaussian, Spectrums_HOPG[i][5:, 0], Spectrums_HOPG[i][5:, 1], p0=p0, bounds = bounds, method = 'dogbox') # Fit the data
    amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, cte = params
    #print(f"Parameters for the fit of {Spectrum_names_HOPG[i]}: \n amplitude1 = {amplitude1} \n mean1 = {mean1} \n stddev1 = {stddev1} \n amplitude2 = {amplitude2} \n mean2 = {mean2} \n stddev2 = {stddev2} \n cte = {cte} \n")
    errors = np.sqrt(np.diag(covariance)) # Standard deviation errors on the parameters

    HOPG_K_alpha.append(mean1)
    HOPG_K_beta.append(mean2)   
    HOPG_delta_K_alpha.append(stddev1)
    HOPG_delta_K_beta.append(stddev2)

    plt.figure(figsize=(8, 6))
    plt.plot(Spectrums_HOPG[i][:, 0], Spectrums_HOPG[i][:, 1], label='data', color = 'darkslategrey')
    plt.plot(Spectrums_HOPG[i][:, 0], double_gaussian(Spectrums_HOPG[i][:, 0], *params), label='fit', color = 'darkorange')
    plt.legend()
    plt.grid()
    plt.title(f"{Spectrum_names_HOPG[i]} fitted with a double gaussian")
    plt.xlabel('\u03B2 (°)')
    plt.ylabel('R(1/s)')
    plt.text(0.1, 0.9, f'Amplitude_\u03B1: {amplitude1:.2f} +- {"{:.5f}".format(errors[0])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.85, f'Mean_\u03B1: {mean1:.2f} +- {"{:.5f}".format(errors[1])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.8, f'Std_\u03B1: {stddev1:.2f} +- {"{:.5f}".format(errors[2])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.75, f'Amplitude_\u03B2: {amplitude2:.2f} +- {"{:.5f}".format(errors[3])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.7, f'Mean_\u03B2: {mean2:.2f} +- {"{:.5f}".format(errors[4])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.65, f'Std_\u03B2: {stddev2:.2f} +- {"{:.5f}".format(errors[5])}', transform=plt.gca().transAxes, color='black')
    plt.savefig(f"images/{Spectrum_names_HOPG[i]}_fitted.png")
    #plt.show()
    plt.close()

##### Calculate the Resolutive Power
HOPG_Potencia_Resolutiva_alpha = Potencia_Resolutiva(HOPG_K_alpha, HOPG_delta_K_alpha)
HOPG_Potencia_Resolutiva_beta = Potencia_Resolutiva(HOPG_K_beta[1:], HOPG_delta_K_beta[1:])

print(f"Resolutive Power for HOPG K_alpha: {HOPG_Potencia_Resolutiva_alpha}")
print(f"Resolutive Power for HOPG K_beta: {HOPG_Potencia_Resolutiva_beta}")

##### Calculate the d-spacing
n = np.array([1,2,3])
HOPG_d_alpha = d_calculator(n, lambda_K_alpha, np.array(HOPG_K_alpha))
HOPG_d_beta = d_calculator(n[1:], lambda_K_beta, np.array(HOPG_K_beta[1:]))
d_HOPG = np.mean(np.concatenate((HOPG_d_alpha, HOPG_d_beta)))
print(f"d-spacing for HOPG: {d_HOPG}\n")


################################# Safira (lab 6) ###############################################################
##### Draw the graphs
Safira_spectrum = reader("Safira_spectrum.xry", 10, 45, 0.1, 1)
plotter(Safira_spectrum, "Safira Spectrum")

Safira_1stGroup = reader("Safira_1st_Group.xry", 17, 21, 0.1, 5)
plotter(Safira_1stGroup, "Safira 1st Group")

Safira_2ndGroup = reader("Safira_2nd_Group.xry", 26, 31, 0.1, 5)
plotter(Safira_2ndGroup, "Safira 2nd Group")

###### Obtain the K_alpha and K_beta peaks
Spectrums_Safira = [Safira_1stGroup, Safira_2ndGroup]
Spectrum_names_Safira = ['Safira_1stGroup', 'Safira_2ndGroup']
Init_Guess_Safira = [[350, 20, 0.5, 100, 17.7, 0.5, 100],
                     [150, 30, 0.5, 50, 26.75, 0.5, 50]]
Bounds_Safira = [([300, 19.7, 0, 50, 17.5, 0, 75], [450, 20.25, 1, 150, 18, 1, 125]),
                    ([120, 29.5, 0, 20, 26, 0, 20], [200, 30.75, 1, 75, 27, 1, 80])]

for i in range(len(Spectrums_Safira)):
    p0 = Init_Guess_Safira[i]  # Initial guess for the parameters
    bounds = Bounds_Safira[i]  # Lower and upper bounds on parameters

    params, covariance = curve_fit(double_gaussian, Spectrums_Safira[i][:, 0], Spectrums_Safira[i][:, 1], p0=p0, bounds = bounds) # Fit the data
    amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, cte = params
    #print(f"Parameters for the fit of {Spectrum_names_Safira[i]}: \n amplitude1 = {amplitude1} \n mean1 = {mean1} \n stddev1 = {stddev1} \n amplitude2 = {amplitude2} \n mean2 = {mean2} \n stddev2 = {stddev2} \n cte = {cte} \n")
    errors = np.sqrt(np.diag(covariance)) # Standard deviation errors on the parameters
    
    Safira_K_alpha.append(mean1)
    Safira_K_beta.append(mean2)
    Safira_delta_K_alpha.append(stddev1)
    Safira_delta_K_beta.append(stddev2)

    plt.figure(figsize=(8, 6))
    plt.plot(Spectrums_Safira[i][:, 0], Spectrums_Safira[i][:, 1], label='data', color = 'darkmagenta')
    plt.plot(Spectrums_Safira[i][:, 0], double_gaussian(Spectrums_Safira[i][:, 0], *params), label='fit', color = 'darkorange')
    plt.legend()
    plt.grid()
    plt.title(f"{Spectrum_names_Safira[i]} fitted with a double gaussian")
    plt.xlabel('\u03B2 (°)')
    plt.ylabel('R(1/s)')
    plt.text(0.1, 0.9, f'Amplitude_\u03B1: {amplitude1:.2f} +- {"{:.5f}".format(errors[0])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.85, f'Mean_\u03B1: {mean1:.2f} +- {"{:.5f}".format(errors[1])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.8, f'Std_\u03B1: {stddev1:.2f} +- {"{:.5f}".format(errors[2])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.75, f'Amplitude_\u03B2: {amplitude2:.2f} +- {"{:.5f}".format(errors[1])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.7, f'Mean_\u03B2: {mean2:.2f} +- {"{:.5f}".format(errors[4])}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.65, f'Std_\u03B2: {stddev2:.2f} +- {"{:.5f}".format(errors[5])}', transform=plt.gca().transAxes, color='black')
    plt.savefig(f"images/{Spectrum_names_Safira[i]}_fitted.png")
    #plt.show()
    plt.close()

###### Calculate the Resolutive Power
Safira_Potencia_Resolutiva_alpha = Potencia_Resolutiva(Safira_K_alpha, Safira_delta_K_alpha)
Safira_Potencia_Resolutiva_beta = Potencia_Resolutiva(Safira_K_beta, Safira_delta_K_beta)

print(f"Resolutive Power for Safira K_alpha: {Safira_Potencia_Resolutiva_alpha}")
print(f"Resolutive Power for Safira K_beta: {Safira_Potencia_Resolutiva_beta}")

###### Calculate the d-spacing
n = np.array([1,2])
Safira_d_alpha = d_calculator(n, lambda_K_alpha, np.array(Safira_K_alpha))
Safira_d_beta = d_calculator(n, lambda_K_beta, np.array(Safira_K_beta))

d_Safira = np.mean(np.concatenate((Safira_d_alpha, Safira_d_beta)))
print(f"d-spacing for Safira: {d_Safira} \n")
