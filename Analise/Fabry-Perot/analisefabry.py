# from read_lab_file import read_lab_file  # Import the reader function from your reader file
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# Definir a função gaussiana
def gauss(x, A, mu, sigma, d):
    return (A * np.exp(0.5* -(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2) + d

### aula3_4.lab ###

with open('dados_para_gauss/aula3_4.lab', 'r') as file:
    lines = file.readlines()

data34 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data34.append(values)

df34 = pd.DataFrame(data34, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo34 = np.array(df34['T.Piezo'])
T_Foto34 = np.array(df34['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados = T_Piezo34[(T_Piezo34 >= 3) & (T_Piezo34 <= 4)]

# Calcular a média e o desvio padrão dos dados selecionados
mean34 = np.mean(valores_selecionados)
std_dev34 = np.std(valores_selecionados)

# Ajustar a gaussiana apenas para o intervalo selecionado
p034 = [T_Foto34.max(), mean34, std_dev34, 5]  # Estimativas iniciais para A, mu e sigma
popt34, covariance = curve_fit(gauss, valores_selecionados, T_Foto34[(T_Piezo34 >= 3) & (T_Piezo34 <= 4)], p0=p034)
a_error, mu_error, sigma_error, d_error = np.sqrt(np.diag(covariance))

x_fit34 = np.linspace(3, 4)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo34, T_Foto34, color='blue')
plt.plot(x_fit34, gauss(x_fit34, *popt34), color='red', label=r'$A = %.3f \pm %.3f ; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt34[0], a_error, popt34[1],mu_error,popt34[2],sigma_error, popt34[3], d_error))
plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula34')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show()

### aula3_5.lab ###

with open('dados_para_gauss/aula3_5.lab', 'r') as file:
    lines = file.readlines()

data35 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data35.append(values)

df35 = pd.DataFrame(data35, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo35 = np.array(df35['T.Piezo'])
T_Foto35 = np.array(df35['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados = T_Piezo35[(T_Piezo35 >= 3) & (T_Piezo35 <= 4)]

# Calcular a média e o desvio padrão dos dados selecionados
mean35 = np.mean(valores_selecionados)
std_dev35 = np.std(valores_selecionados)

# Ajustar a gaussiana apenas para o intervalo selecionado
p035 = [T_Foto35.max(), mean35, std_dev35, 5]  # Estimativas iniciais para A, mu e sigma
popt35, covariance = curve_fit(gauss, valores_selecionados, T_Foto35[(T_Piezo35 >= 3) & (T_Piezo35 <= 4)], p0=p035)

a_error, mu_error, sigma_error, d_error = np.sqrt(np.diag(covariance))
x_fit35 = np.linspace(3, 4.5, 100)  # Intervalo para o ajuste
a_error, mu_error, sigma_error, d_error = np.sqrt(np.diag(covariance))
# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo35, T_Foto35, color='black')

plt.plot(x_fit35, gauss(x_fit35, *popt35), color='red', label=r'$A = %.3f \pm %.3f ; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt35[0], a_error, popt35[1],mu_error,popt35[2],sigma_error, popt35[3], d_error))
plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula35')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() 

### aula3_6.lab ###

with open('dados_para_gauss/aula3_6.lab', 'r') as file:
    lines = file.readlines()

data36 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data36.append(values)

df36 = pd.DataFrame(data36, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])
rolling_mean_foto = df36['T.Foto'].rolling(window=10).mean()

T_Piezo36 = np.array(df36['T.Piezo'])
T_Foto36 = np.array(rolling_mean_foto)


nan_indices = np.argwhere(np.isnan(T_Foto36))

for i in nan_indices:
    T_Foto36[i] = df36['T.Foto'][i]

T_Foto36 = df36['T.Foto']
valores_selecionados_1 = T_Piezo36[(T_Piezo36 >= 4) & (T_Piezo36 <= 5)]
valores_selecionados_2 = T_Piezo36[(T_Piezo36 >= 5) & (T_Piezo36 <= 6)]

# Calcular a média e o desvio padrão dos dados selecionados
mean36_1 = np.mean(valores_selecionados_1)
mean36_2 = np.mean(valores_selecionados_2)
std_dev36_1 = np.std(valores_selecionados_1) 
std_dev36_2 = np.std(valores_selecionados_2)

# Definir a função gaussiana
def gauss_sum(x, A, mu1, sigma1, B, mu2, sigma2,d):
    return (A * np.exp(0.5* -(x - mu1) ** 2 / (sigma1 ** 2)))/np.sqrt(2*np.pi*sigma1**2) + (B * np.exp(0.5*-(x - mu2) ** 2 / (sigma2 ** 2)))/np.sqrt(2*np.pi*sigma2**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p036 = [T_Foto36.max(), mean36_1, std_dev36_1,T_Foto36[(T_Piezo36 >= 5) & (T_Piezo36 <= 6)].max(), mean36_2, std_dev36_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt36_1, covariance = curve_fit(gauss_sum, T_Piezo36, T_Foto36, p0=p036)
x_fit36 = np.linspace(4, 6)  # Intervalo para o ajuste
x_fit36_2 = np.linspace(5, 6, 100)
a1_error, mu1_error, sigma1_error, a2_error, mu2_error, sigma2_error, d_error = np.sqrt(np.diag(covariance))
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt36_1[0] , popt36_1[1], popt36_1[2] , popt36_1[3], popt36_1[4], popt36_1[5], popt36_1[6]

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo36, T_Foto36)
#plt.plot(, label=r'$%3.f \cdot \frac{1}{\sqrt{2\pi%3.f}}e^{\frac{-(x-%.3f)^2}{%.3f^2}}$,  $\mu_2$ = %.3f, $\sigma_2$ = %.3f, d = %.3f' % (A1, sigma1, mu1, sigma1, mu2, sigma2, d))

plt.plot(x_fit36, gauss_sum(x_fit36, *popt36_1), label=r'$A1 = %.3f \pm %.3f ; \mu 1 = %.3f\pm %.3f ; \sigma 1 = %.3f\pm %.3f ; A2 = %.3f\pm %.3f ; \mu 2 = %.3f\pm %.3f ; \sigma 2 = %.3f\pm %.3f ; d = %.3f\pm %.3f $' %(A1,a1_error,mu1,mu1_error,sigma1,sigma1_error,A2,a2_error,mu2,mu2_error,sigma2,sigma2_error,d,d_error))


plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula36 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='large')
plt.show() 
print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt36_1[0])
print('mu1 = ', popt36_1[1])
print('sigma1 = ', popt36_1[2])
print('A2 = ', popt36_1[3])
print('mu2 = ', popt36_1[4])
print('sigma2 = ', popt36_1[5])
print('d = ', popt36_1[6])
print('\n')

### aula3_7.lab ###

with open('dados_para_gauss/aula3_7.lab', 'r') as file:
    lines = file.readlines()

data37 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data37.append(values)

df37 = pd.DataFrame(data37, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo37 = np.array(df37['T.Piezo'])
T_Foto37 = np.array(df37['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo37[(T_Piezo37 >= 4) & (T_Piezo37 <= 5)]
valores_selecionados_2 = T_Piezo37[(T_Piezo37 >= 5) & (T_Piezo37 <= 6)]


# Calcular a média e o desvio padrão dos dados selecionados
mean37_1 = np.mean(valores_selecionados_1)
std_dev37_1 = np.std(valores_selecionados_1)
mean37_2 = np.mean(valores_selecionados_2)
std_dev37_2 = np.std(valores_selecionados_2)


# Definir a função gaussiana
def gauss(x, A, mu, sigma, d):
    return (A * np.exp(0.5* -(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p037_1 = [T_Foto37.max(), mean37_1, std_dev37_1, 5]  # Estimativas iniciais para A, mu e sigma
popt37_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto37[(T_Piezo37 >= 4) & (T_Piezo37 <= 5)], p0=p037_1)
a1_error, mu1_error, sigma1_error, d_error = np.sqrt(np.diag(covariance1))
p037_2 = [T_Foto37.max(), 5.7, 0.1, 5]  # Estimativas iniciais para A, mu e sigma
popt37_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto37[(T_Piezo37 >= 5) & (T_Piezo37 <= 6)], p0=p037_2)
a2_error, mu2_error, sigma2_error, d_error = np.sqrt(np.diag(covariance2))

x_fit37_1 = np.linspace(4, 4.95)  # Intervalo para o ajuste
x_fit37_2 = np.linspace(5, 6.5)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo37, T_Foto37, color='black')

plt.plot(x_fit37_1, gauss(x_fit37_1, *popt37_1), label=r'$A = %.3f \pm %.3f ; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt37_1[0], a1_error, popt37_1[1],mu1_error,popt37_1[2],sigma1_error, popt37_1[3], d_error))

plt.plot(x_fit37_2, gauss(x_fit37_2, *popt37_2), label=r'$A = %.3f \pm %.3f ; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt37_2[0], a2_error, popt37_2[1],mu2_error,popt37_2[2],sigma2_error, popt37_2[3], d_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula37 ')

x_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt37_1[0])
print('mu1 = ', popt37_1[1])
print('sigma1 = ', popt37_1[2])
print('A2 = ', popt37_2[0])
print('mu2 = ', popt37_2[1])
print('sigma2 = ', popt37_2[2])
print('d = ', popt37_2[3])
print('\n') 

### aula3_8.lab ###

with open('dados_para_gauss/aula3_8.lab', 'r') as file:
    lines = file.readlines()

data38 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data38.append(values)

df38 = pd.DataFrame(data38, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo38 = np.array(df38['T.Piezo'])
T_Foto38 = np.array(df38['T.Foto'])

valores_selecionados_1 = T_Piezo38[(T_Piezo38 >= 4) & (T_Piezo38 <= 5)]
valores_selecionados_2 = T_Piezo38[(T_Piezo38 >= 6) & (T_Piezo38 <= 7)]

# Calcular a média e o desvio padrão dos dados selecionados
mean38_1 = np.mean(valores_selecionados_1)
mean38_2 = np.mean(valores_selecionados_2)
std_dev38_1 = np.std(valores_selecionados_1) 
std_dev38_2 = np.std(valores_selecionados_2)

# Definir a função gaussiana
def gauss_sum(x, A, mu1, sigma1, B, mu2, sigma2,d):
    return (A * np.exp(0.5* -(x - mu1) ** 2 / (sigma1 ** 2)))/np.sqrt(2*np.pi*sigma1**2) + (B * np.exp(0.5*-(x - mu2) ** 2 / (sigma2 ** 2)))/np.sqrt(2*np.pi*sigma2**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p038 = [T_Foto38.max(), mean38_1, std_dev38_1,T_Foto38[(T_Piezo38 >= 6) & (T_Piezo38 <= 7)].max(), mean38_2, std_dev38_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt38_1, covariance = curve_fit(gauss_sum, T_Piezo38, T_Foto38, p0=p038)
x_fit38 = np.linspace(4, 7)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt38_1[0] , popt38_1[1], popt38_1[2] , popt38_1[3], popt38_1[4], popt38_1[5], popt38_1[6]
a1_error, mu1_error, sigma1_error, a2_error, mu2_error, sigma2_error, d_error = np.sqrt(np.diag(covariance))
# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo38, T_Foto38)
#plt.plot(, label=r'$%3.f \cdot \frac{1}{\sqrt{2\pi%3.f}}e^{\frac{-(x-%.3f)^2}{%.3f^2}}$,  $\mu_2$ = %.3f, $\sigma_2$ = %.3f, d = %.3f' % (A1, sigma1, mu1, sigma1, mu2, sigma2, d))

plt.plot(x_fit38, gauss_sum(x_fit38, *popt38_1), label=r'$A1 = %.3f \pm %.3f ; \mu 1 = %.3f\pm %.3f ; \sigma 1 = %.3f\pm %.3f ; A2 = %.3f\pm %.3f ; \mu 2 = %.3f\pm %.3f ; \sigma 2 = %.3f\pm %.3f ; d = %.3f\pm %.3f $' %(A1,a1_error,mu1,mu1_error,sigma1,sigma1_error,A2,a2_error,mu2,mu2_error,sigma2,sigma2_error,d,d_error))
plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula38 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='large')
plt.show() 
print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt38_1[0])
print('mu1 = ', popt38_1[1])
print('sigma1 = ', popt38_1[2])
print('A2 = ', popt38_1[3])
print('mu2 = ', popt38_1[4])
print('sigma2 = ', popt38_1[5])
print('d = ', popt38_1[6])
print('\n')

### aula3_12.lab ###

with open('dados_para_gauss/aula3_12.lab', 'r') as file:
    lines = file.readlines()

data312 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data312.append(values)

df312 = pd.DataFrame(data312, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo312 = np.array(df312['T.Piezo'])
T_Foto312 = np.array(df312['T.Foto'])

valores_selecionados_1 = T_Piezo312[(T_Piezo312 >= 4) & (T_Piezo312 <= 5)]
valores_selecionados_2 = T_Piezo312[(T_Piezo312 >= 6) & (T_Piezo312 <= 7)]
valores_selecionados_3 = T_Piezo312[(T_Piezo312 >= 0) & (T_Piezo312 <= 1)]
valores_selecionados_4 = T_Piezo312[(T_Piezo312 >= 1) & (T_Piezo312 <= 2)]


# Calcular a média e o desvio padrão dos dados selecionados
mean312_1 = np.mean(valores_selecionados_1)
mean312_2 = np.mean(valores_selecionados_2)
std_dev312_1 = np.std(valores_selecionados_1) 
std_dev312_2 = np.std(valores_selecionados_2)
mean312_3 = np.mean(valores_selecionados_3)
std_dev312_3 = np.std(valores_selecionados_3) 
mean312_4 = np.mean(valores_selecionados_4)
std_dev312_4 = np.std(valores_selecionados_4) 

# Definir a função gaussiana
def gauss_sum(x, A, mu1, sigma1, B, mu2, sigma2,d):
    return (A * np.exp(0.5* -(x - mu1) ** 2 / (sigma1 ** 2)))/np.sqrt(2*np.pi*sigma1**2) + (B * np.exp(0.5*-(x - mu2) ** 2 / (sigma2 ** 2)))/np.sqrt(2*np.pi*sigma2**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p0312 = [T_Foto312.max(), mean312_1, std_dev312_1,T_Foto312[(T_Piezo312 >= 6) & (T_Piezo312 <= 7)].max(), mean312_2, std_dev312_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt312_1, covariance3_4 = curve_fit(gauss_sum, T_Piezo312, T_Foto312, p0=p0312)
x_fit312_1 = np.linspace(4, 7.5)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d3_4 = popt312_1[0] , popt312_1[1], popt312_1[2] , popt312_1[3], popt312_1[4], popt312_1[5], popt312_1[6]
a3_error, mu3_error, sigma3_error, a4_error,mu4_error,sigma4_error,d3_4_error = np.sqrt(np.diag(covariance3_4))

p0312_2 = [T_Foto312.max(), mean312_3, std_dev312_3,T_Foto312[(T_Piezo312 >= 1) & (T_Piezo312 <= 2)].max(), 2, 0.05, 0.5]  # Estimativas iniciais para A, mu1 e sigma1
popt312_2, covariance1_2 = curve_fit(gauss_sum, T_Piezo312, T_Foto312, p0=p0312_2)
x_fit312_2 = np.linspace(0, 1.8)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d1_2 = popt312_2[0] , popt312_2[1], popt312_2[2] , popt312_2[3], popt312_2[4], popt312_2[5], popt312_2[6]
a1_error, mu1_error, sigma1_error, a2_error,mu2_error,sigma2_error,d1_2_error = np.sqrt(np.diag(covariance1_2))

# p0312_2 = [T_Foto312.max(), mean312_3, std_dev312_3, 5]  # Estimativas iniciais para A, mu e sigma
# popt312_2, _ = curve_fit(gauss, valores_selecionados_3, T_Foto312[(T_Piezo312 >= 0) & (T_Piezo312 <= 1)], p0=p0312_2)
# x_fit312_2 = np.linspace(0, 1)  # Intervalo para o ajuste

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo312, T_Foto312)

plt.plot(x_fit312_1, gauss_sum(x_fit312_1, *popt312_1), label=r'$A3 = %.3f \pm %.3f; \mu 3 = %.3f \pm %.3f; \sigma 3 = %.3f \pm %.3f; A4= %.3f \pm %.3f; \mu 4 = %.3f \pm %.3f; \sigma 4 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A3,a3_error,mu3,mu3_error,sigma3,sigma3_error,A4,a4_error,mu4,mu4_error,sigma4,sigma4_error,d3_4,d3_4_error))
plt.plot(x_fit312_2, gauss_sum(x_fit312_2, *popt312_2), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 1 = %.3f \pm %.3f; A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A1,a1_error,mu1,mu1_error,sigma1,sigma1_error,A2,a2_error,mu2,mu2_error,sigma2,sigma2_error,d1_2,d1_2_error))

plt.xlabel('T.Piezo')
plt.legend(fontsize='large')
plt.show() 
plt.ylabel('T.Foto')
plt.title('aula312 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='large')
plt.show() 
print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt312_2[0])
print('mu1 = ', popt312_2[1])
print('sigma1 = ', popt312_2[2])

print('A2 = ', popt312_2[3])
print('mu2 = ', popt312_2[4])
print('sigma2 = ', popt312_2[5])

print('A3 = ', popt312_1[0])
print('mu3 = ', popt312_1[1])
print('sigma3 = ', popt312_1[2])

print('A4 = ', popt312_1[3])
print('mu4 = ', popt312_1[4])
print('sigma4 = ', popt312_1[5])
print('d = ', popt312_1[6])
print('\n')


### aula3_14.lab ###

with open('dados_para_gauss/aula3_14.lab', 'r') as file:
    lines = file.readlines()

data314 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data314.append(values)

df314 = pd.DataFrame(data314, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo314 = np.array(df314['T.Piezo'])
T_Foto314 = np.array(df314['T.Foto'])

valores_selecionados_1 = T_Piezo314[(T_Piezo314 >= 4) & (T_Piezo314 <= 5)]
valores_selecionados_2 = T_Piezo314[(T_Piezo314 >= 6) & (T_Piezo314 <= 7)]
valores_selecionados_3 = T_Piezo314[(T_Piezo314 >= 0) & (T_Piezo314 <= 1)]
valores_selecionados_4 = T_Piezo314[(T_Piezo314 >= 1) & (T_Piezo314 <= 2)]


# Calcular a média e o desvio padrão dos dados selecionados
mean314_1 = np.mean(valores_selecionados_1)
mean314_2 = np.mean(valores_selecionados_2)
std_dev314_1 = np.std(valores_selecionados_1) 
std_dev314_2 = np.std(valores_selecionados_2)
mean314_3 = np.mean(valores_selecionados_3)
std_dev314_3 = np.std(valores_selecionados_3) 
mean314_4 = np.mean(valores_selecionados_4)
std_dev314_4 = np.std(valores_selecionados_4) 

# Definir a função gaussiana
def gauss_sum(x, A, mu1, sigma1, B, mu2, sigma2,d):
    return (A * np.exp(0.5* -(x - mu1) ** 2 / (sigma1 ** 2)))/np.sqrt(2*np.pi*sigma1**2) + (B * np.exp(0.5*-(x - mu2) ** 2 / (sigma2 ** 2)))/np.sqrt(2*np.pi*sigma2**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p0314 = [T_Foto314.max(), mean314_1, std_dev314_1,T_Foto314[(T_Piezo314 >= 6) & (T_Piezo314 <= 7)].max(), mean314_2, std_dev314_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt314_1, covariance1 = curve_fit(gauss_sum, T_Piezo314, T_Foto314, p0=p0314)
x_fit314_1 = np.linspace(5, 7.5)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d3_4 = popt314_1[0] , popt314_1[1], popt314_1[2] , popt314_1[3], popt314_1[4], popt314_1[5], popt314_1[6]
a3_error, mu3_error, sigma3_error, a4_error,mu4_error,sigma4_error,d3_4_error = np.sqrt(np.diag(covariance1))


p0314_2 = [T_Foto314.max(), mean314_3, std_dev314_3,T_Foto314[(T_Piezo314 >= 1) & (T_Piezo314 <= 2)].max(), 2, 0.05, 0.5]  # Estimativas iniciais para A, mu1 e sigma1
popt314_2, covariance2 = curve_fit(gauss_sum, T_Piezo314, T_Foto314, p0=p0314_2)
x_fit314_2 = np.linspace(0, 1.8)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d1_2 = popt314_2[0] , popt314_2[1], popt314_2[2] , popt314_2[3], popt314_2[4], popt314_2[5], popt314_2[6]
a2_error, mu2_error, sigma2_error, a1_error,mu1_error,sigma1_error,d1_2_error = np.sqrt(np.diag(covariance2))

# p0312_2 = [T_Foto312.max(), mean312_3, std_dev312_3, 5]  # Estimativas iniciais para A, mu e sigma
# popt312_2, _ = curve_fit(gauss, valores_selecionados_3, T_Foto312[(T_Piezo312 >= 0) & (T_Piezo312 <= 1)], p0=p0312_2)
# x_fit312_2 = np.linspace(0, 1)  # Intervalo para o ajuste

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo314, T_Foto314)

plt.plot(x_fit314_1, gauss_sum(x_fit314_1, *popt314_1), label=r'$A3 = %.3f \pm %.3f; \mu 3 = %.3f \pm %.3f; \sigma 3 = %.3f \pm %.3f; A4 = %.3f \pm %.3f; \mu 4 = %.3f \pm %.3f; \sigma 4  = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A3,a3_error,mu3,mu3_error,sigma3,sigma3_error,A4,a4_error,mu4,mu4_error,sigma4,sigma4_error,d3_4,d3_4_error))
plt.plot(x_fit314_2, gauss_sum(x_fit314_2, *popt314_2), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 1 = %.3f \pm %.3f; A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A1,a1_error,mu1,mu1_error,sigma1,sigma1_error,A2,a2_error,mu2,mu2_error,sigma2,sigma2_error,d1_2,d1_2_error))

plt.xlabel('T.Piezo')
plt.legend(fontsize='large')
plt.show() 
plt.ylabel('T.Foto')
plt.title('aula314 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='large')
plt.show() 
print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt314_2[0])
print('mu1 = ', popt314_2[1])
print('sigma1 = ', popt314_2[2])

print('A2 = ', popt314_2[3])
print('mu2 = ', popt314_2[4])
print('sigma2 = ', popt314_2[5])

print('A3 = ', popt314_1[0])
print('mu3 = ', popt314_1[1])
print('sigma3 = ', popt314_1[2])

print('A4 = ', popt314_1[3])
print('mu4 = ', popt314_1[4])
print('sigma4 = ', popt314_1[5])
print('d = ', popt314_1[6])
print('\n')

### aula3_16.lab ###

with open('dados_para_gauss/aula3_16.lab', 'r') as file:
    lines = file.readlines()

data316 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data316.append(values)

df316 = pd.DataFrame(data316, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo316 = np.array(df316['T.Piezo'])
T_Foto316 = np.array(df316['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo316[(T_Piezo316 >= 0) & (T_Piezo316 <= 1.5)]
valores_selecionados_2 = T_Piezo316[(T_Piezo316 >= 6) & (T_Piezo316 <= 8)]


# Calcular a média e o desvio padrão dos dados selecionados
mean316_1 = np.mean(valores_selecionados_1)
std_dev316_1 = np.std(valores_selecionados_1)
mean316_2 = np.mean(valores_selecionados_2)
std_dev316_2 = np.std(valores_selecionados_2)


# Definir a função gaussiana
def gauss(x, A, mu, sigma, d):
    return (A * np.exp(0.5* -(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p0316_1 = [T_Foto316.max(), mean316_1, std_dev316_1, 5]  # Estimativas iniciais para A, mu e sigma
popt316_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto316[(T_Piezo316 >= 0) & (T_Piezo316 <= 1.5)], p0=p0316_1)
a1_error, mu1_error, sigma1_error, d1_error = np.sqrt(np.diag(covariance1))

p0316_2 = [T_Foto316.max(), mean316_2, std_dev316_2, 5]  # Estimativas iniciais para A, mu e sigma
popt316_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto316[(T_Piezo316 >= 6) & (T_Piezo316 <= 8)], p0=p0316_2)
a2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

x_fit316_1 = np.linspace(0, 1.5)  # Intervalo para o ajuste
x_fit316_2 = np.linspace(6, 8)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo316, T_Foto316)

plt.plot(x_fit316_1, gauss(x_fit316_1, *popt316_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 1 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt316_1[0], a1_error, popt316_1[1], mu1_error, popt316_1[2], sigma1_error, popt316_1[3], d1_error))
plt.plot(x_fit316_2, gauss(x_fit316_2, *popt316_2), label=r'$A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt316_2[0], a2_error, popt316_2[1], mu2_error, popt316_2[2], sigma2_error, popt316_2[3], d2_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula316 ')

x_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt316_1[0])
print('mu1 = ', popt316_1[1])
print('sigma1 = ', popt316_1[2])
print('A2 = ', popt316_2[0])
print('mu2 = ', popt316_2[1])
print('sigma2 = ', popt316_2[2])
print('d = ', popt316_2[3])
print('\n') 

### aula3_17.lab ###

with open('dados_para_gauss/aula3_17.lab', 'r') as file:
    lines = file.readlines()

data317 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data317.append(values)

df317 = pd.DataFrame(data317, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo317 = np.array(df317['T.Piezo'])
T_Foto317 = np.array(df317['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo317[(T_Piezo317 >= 0) & (T_Piezo317 <= 2)]
valores_selecionados_2 = T_Piezo317[(T_Piezo317 >= 6) & (T_Piezo317 <= 8.5)]


# Calcular a média e o desvio padrão dos dados selecionados
mean317_1 = np.mean(valores_selecionados_1)
std_dev317_1 = np.std(valores_selecionados_1)
mean317_2 = np.mean(valores_selecionados_2)
std_dev317_2 = np.std(valores_selecionados_2)


# Definir a função gaussiana
def gauss(x, A, mu, sigma, d):
    return (A * np.exp(0.5* -(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p0317_1 = [T_Foto317.max(), mean317_1, std_dev317_1, 5]  # Estimativas iniciais para A, mu e sigma
popt317_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto317[(T_Piezo317 >= 0) & (T_Piezo317 <= 2)], p0=p0317_1)
a1_error, mu1_error, sigma1_error, d1_error = np.sqrt(np.diag(covariance1))

p0317_2 = [T_Foto317.max(), mean317_2, std_dev317_2, 5]  # Estimativas iniciais para A, mu e sigma
popt317_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto317[(T_Piezo317 >= 6) & (T_Piezo317 <= 8.5)], p0=p0317_2)
a2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

x_fit317_1 = np.linspace(0, 2)  # Intervalo para o ajuste
x_fit317_2 = np.linspace(6, 8.5)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo317, T_Foto317)

plt.plot(x_fit317_1, gauss(x_fit317_1, *popt317_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 1 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt317_1[0], a1_error, popt317_1[1], mu1_error, popt317_1[2], sigma1_error, popt317_1[3], d1_error))
plt.plot(x_fit317_2, gauss(x_fit317_2, *popt317_2), label=r'$A2 = %.3f \pm %.3f; \mu 2  = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt317_2[0], a2_error, popt317_2[1], mu2_error, popt317_2[2], sigma2_error, popt317_2[3], d2_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula317 ')

x_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt317_1[0])
print('mu1 = ', popt317_1[1])
print('sigma1 = ', popt317_1[2])
print('A2 = ', popt317_2[0])
print('mu2 = ', popt317_2[1])
print('sigma2 = ', popt317_2[2])
print('d = ', popt317_2[3])
print('\n') 

### aula3_18.lab ###

with open('dados_para_gauss/aula3_18.lab', 'r') as file:
    lines = file.readlines()

data318 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data318.append(values)

df318 = pd.DataFrame(data318, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo318 = np.array(df318['T.Piezo'])
T_Foto318 = np.array(df318['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo318[(T_Piezo318 >= 0) & (T_Piezo318 <= 2)]
valores_selecionados_2 = T_Piezo318[(T_Piezo318 >= 6) & (T_Piezo318 <= 8.5)]


# Calcular a média e o desvio padrão dos dados selecionados
mean318_1 = np.mean(valores_selecionados_1)
std_dev318_1 = np.std(valores_selecionados_1)
mean318_2 = np.mean(valores_selecionados_2)
std_dev318_2 = np.std(valores_selecionados_2)


# Definir a função gaussiana
def gauss(x, A, mu, sigma, d):
    return (A * np.exp(0.5* -(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p0318_1 = [T_Foto318.max(), mean318_1, std_dev318_1, 5]  # Estimativas iniciais para A, mu e sigma
popt318_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto318[(T_Piezo318 >= 0) & (T_Piezo318 <= 2)], p0=p0318_1)
a1_error, mu1_error, sigma1_error, d1_error = np.sqrt(np.diag(covariance1))

p0318_2 = [T_Foto318.max(), mean318_2, std_dev318_2, 5]  # Estimativas iniciais para A, mu e sigma
popt318_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto318[(T_Piezo318 >= 6) & (T_Piezo318 <= 8.5)], p0=p0318_2)
a2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

x_fit318_1 = np.linspace(0, 2)  # Intervalo para o ajuste
x_fit318_2 = np.linspace(6, 8.5)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo318, T_Foto318)

plt.plot(x_fit318_1, gauss(x_fit318_1, *popt318_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 1 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt318_1[0], a1_error, popt318_1[1], mu1_error, popt318_1[2], sigma1_error, popt318_1[3], d1_error))
plt.plot(x_fit318_2, gauss(x_fit318_2, *popt318_2), label=r'$A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt318_2[0], a2_error, popt318_2[1], mu2_error, popt318_2[2], sigma2_error, popt318_2[3], d2_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula318 ')

x_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt318_1[0])
print('mu1 = ', popt318_1[1])
print('sigma1 = ', popt318_1[2])
print('A2 = ', popt318_2[0])
print('mu2 = ', popt318_2[1])
print('sigma2 = ', popt318_2[2])
print('d = ', popt318_2[3])
print('\n') 

### aula3_19.lab ###

with open('dados_para_gauss/aula3_19.lab', 'r') as file:
    lines = file.readlines()

data319 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data319.append(values)

df319 = pd.DataFrame(data319, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo319 = np.array(df319['T.Piezo'])
T_Foto319 = np.array(df319['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo319[(T_Piezo319 >= 0) & (T_Piezo319 <= 2)]
valores_selecionados_2 = T_Piezo319[(T_Piezo319 >= 6) & (T_Piezo319 <= 8.5)]


# Calcular a média e o desvio padrão dos dados selecionados
mean319_1 = np.mean(valores_selecionados_1)
std_dev319_1 = np.std(valores_selecionados_1)
mean319_2 = np.mean(valores_selecionados_2)
std_dev319_2 = np.std(valores_selecionados_2)


# Definir a função gaussiana
def gauss(x, A, mu, sigma, d):
    return (A * np.exp(0.5* -(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p0319_1 = [T_Foto319.max(), mean319_1, std_dev319_1, 5]  # Estimativas iniciais para A, mu e sigma
popt319_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto319[(T_Piezo319 >= 0) & (T_Piezo319 <= 2)], p0=p0319_1)
a1_error,mu1_error,sigma1_error,d1_error = np.sqrt(np.diag(covariance1))

p0319_2 = [T_Foto319.max(), mean319_2, std_dev319_2, 5]  # Estimativas iniciais para A, mu e sigma
popt319_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto319[(T_Piezo319 >= 6) & (T_Piezo319 <= 8.5)], p0=p0319_2)
a2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

x_fit319_1 = np.linspace(0, 2)  # Intervalo para o ajuste
x_fit319_2 = np.linspace(6, 8.5)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo319, T_Foto319)

plt.plot(x_fit319_1, gauss(x_fit319_1, *popt319_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 1 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt319_1[0], a1_error, popt319_1[1], mu1_error, popt319_1[2], sigma1_error, popt319_1[3], d1_error))
plt.plot(x_fit319_2, gauss(x_fit319_2, *popt319_2), label=r'$A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt319_2[0], a2_error, popt319_2[1], mu2_error, popt319_2[2], sigma2_error, popt319_2[3], d2_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula319 ')

x_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt319_1[0])
print('mu1 = ', popt319_1[1])
print('sigma1 = ', popt319_1[2])
print('A2 = ', popt319_2[0])
print('mu2 = ', popt319_2[1])
print('sigma2 = ', popt319_2[2])
print('d = ', popt319_2[3])
print('\n') 

### aula3_20.lab ###

with open('dados_para_gauss/aula3_20.lab', 'r') as file:
    lines = file.readlines()

data320 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data320.append(values)

df320 = pd.DataFrame(data320, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo320 = np.array(df320['T.Piezo'])
T_Foto320 = np.array(df320['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo320[(T_Piezo320 >= 0) & (T_Piezo320 <= 2)]
valores_selecionados_2 = T_Piezo320[(T_Piezo320 >= 6) & (T_Piezo320 <= 8.5)]


# Calcular a média e o desvio padrão dos dados selecionados
mean320_1 = np.mean(valores_selecionados_1)
std_dev320_1 = np.std(valores_selecionados_1)
mean320_2 = np.mean(valores_selecionados_2)
std_dev320_2 = np.std(valores_selecionados_2)


# Definir a função gaussiana
def gauss(x, A, mu, sigma, d):
    return (A * np.exp(0.5* -(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p0320_1 = [T_Foto320.max(), mean320_1, std_dev320_1, 5]  # Estimativas iniciais para A, mu e sigma
popt320_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto320[(T_Piezo320 >= 0) & (T_Piezo320 <= 2)], p0=p0320_1)
a1_error, mu1_error, sigma1_error, d1_error = np.sqrt(np.diag(covariance1))

p0320_2 = [T_Foto320.max(), mean320_2, std_dev320_2, 5]  # Estimativas iniciais para A, mu e sigma
popt320_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto320[(T_Piezo320 >= 6) & (T_Piezo320 <= 8.5)], p0=p0320_2)
a2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

x_fit320_1 = np.linspace(0, 2)  # Intervalo para o ajuste
x_fit320_2 = np.linspace(6, 8.5)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo320, T_Foto320)

plt.plot(x_fit320_1, gauss(x_fit320_1, *popt320_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 1 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt320_1[0], a1_error, popt320_1[1], mu1_error, popt320_1[2], sigma1_error, popt320_1[3], d1_error))
plt.plot(x_fit320_2, gauss(x_fit320_2, *popt320_2), label=r'$A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt320_2[0], a2_error, popt320_2[1], mu2_error, popt320_2[2], sigma2_error, popt320_2[3], d2_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula320 ')

x_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt320_1[0])
print('mu1 = ', popt320_1[1])
print('sigma1 = ', popt320_1[2])
print('A2 = ', popt320_2[0])
print('mu2 = ', popt320_2[1])
print('sigma2 = ', popt320_2[2])
print('d = ', popt320_2[3])
print('\n') 

### aula3_21.lab ###

with open('dados_para_gauss/aula3_21.lab', 'r') as file:
    lines = file.readlines()

data321 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data321.append(values)

df321 = pd.DataFrame(data321, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo321 = np.array(df321['T.Piezo'])
T_Foto321 = np.array(df321['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo321[(T_Piezo321 >= 0) & (T_Piezo321 <= 2)]
valores_selecionados_2 = T_Piezo321[(T_Piezo321 >= 6) & (T_Piezo321 <= 8.5)]


# Calcular a média e o desvio padrão dos dados selecionados
mean321_1 = np.mean(valores_selecionados_1)
std_dev321_1 = np.std(valores_selecionados_1)
mean321_2 = np.mean(valores_selecionados_2)
std_dev321_2 = np.std(valores_selecionados_2)


# Definir a função gaussiana
def gauss(x, A, mu, sigma, d):
    return (A * np.exp(0.5* -(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2) + d

# Ajustar a gaussiana apenas para o intervalo selecionado
p0321_1 = [T_Foto321.max(), mean321_1, std_dev321_1, 5]  # Estimativas iniciais para A, mu e sigma
popt321_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto321[(T_Piezo321 >= 0) & (T_Piezo321 <= 2)], p0=p0321_1)

p0321_2 = [T_Foto321.max(), mean321_2, std_dev321_2, 5]  # Estimativas iniciais para A, mu e sigma
popt321_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto321[(T_Piezo321 >= 6) & (T_Piezo321 <= 8.5)], p0=p0321_2)
a2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

x_fit321_1 = np.linspace(0, 2)  # Intervalo para o ajuste
x_fit321_2 = np.linspace(6, 8.5)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo321, T_Foto321)

plt.plot(x_fit321_1, gauss(x_fit321_1, *popt321_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 1 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt321_1[0], a1_error, popt321_1[1], mu1_error, popt321_1[2], sigma1_error, popt321_1[3], d1_error))
plt.plot(x_fit321_2, gauss(x_fit321_2, *popt321_2), label=r'$A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt321_2[0], a2_error, popt321_2[1], mu2_error, popt321_2[2], sigma2_error, popt321_2[3], d2_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula321 ')

x_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt321_1[0])
print('mu1 = ', popt321_1[1])
print('sigma1 = ', popt321_1[2])
print('A2 = ', popt321_2[0])
print('mu2 = ', popt321_2[1])
print('sigma2 = ', popt321_2[2])
print('d = ', popt321_2[3])
print('\n') 