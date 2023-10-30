# from read_lab_file import read_lab_file  # Import the reader function from your reader file
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

### aula3_4.lab ###
'''
with open('aula3_4.lab', 'r') as file:
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

# Definir a função gaussiana
def gauss(x, A, mu, sigma):
    return (A * np.exp(-(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2)

# Ajustar a gaussiana apenas para o intervalo selecionado
p034 = [T_Foto34.max(), mean34, std_dev34]  # Estimativas iniciais para A, mu e sigma
popt34, _ = curve_fit(gauss, valores_selecionados, T_Foto34[(T_Piezo34 >= 3) & (T_Piezo34 <= 4)], p0=p034)

x_fit34 = np.linspace(2.5, 4.5, 100)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo34, T_Foto34, color='blue')

plt.plot(x_fit34, gauss(x_fit34, *popt34), color='red', label='Distribuição Normal (Gaussiana) Ajustada')
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

with open('aula3_5.lab', 'r') as file:
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

# Definir a função gaussiana
def gauss(x, A, mu, sigma):
    return (A * np.exp(-(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2)

# Ajustar a gaussiana apenas para o intervalo selecionado
p035 = [T_Foto35.max(), mean35, std_dev35]  # Estimativas iniciais para A, mu e sigma
popt35, _ = curve_fit(gauss, valores_selecionados, T_Foto35[(T_Piezo35 >= 3) & (T_Piezo35 <= 4)], p0=p035)

x_fit35 = np.linspace(3, 4.5, 100)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo35, T_Foto35, color='black')

plt.plot(x_fit35, gauss(x_fit35, *popt35), color='red', label='Distribuição Normal (Gaussiana) Ajustada')
plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula35 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() '''

### aula3_6.lab ###

with open('aula3_6.lab', 'r') as file:
    lines = file.readlines()

data36 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data36.append(values)

df36 = pd.DataFrame(data36, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo36 = np.array(df36['T.Piezo'])
T_Foto36 = np.array(df36['T.Foto'])

valores_selecionados_1 = T_Piezo36[(T_Piezo36 >= 4) & (T_Piezo36 <= 5)]
valores_selecionados_2 = T_Piezo36[(T_Piezo36 >= 5) & (T_Piezo36 <= 6)]

# Calcular a média e o desvio padrão dos dados selecionados
mean36_1 = np.mean(valores_selecionados_1)
mean36_2 = np.mean(valores_selecionados_2)
std_dev36_1 = np.std(valores_selecionados_1)
std_dev36_2 = np.std(valores_selecionados_2)

# Definir a função gaussiana
def gauss_sum(x1, A, mu1, sigma1, x2, B, mu2, sigma2):
    return (A * np.exp(-(x1 - mu1) ** 2 / (sigma1 ** 2)))/np.sqrt(2*np.pi*sigma1**2) + (B * np.exp(-(x2 - mu2) ** 2 / (sigma2 ** 2)))/np.sqrt(2*np.pi*sigma2**2)

# Ajustar a gaussiana apenas para o intervalo selecionado
p036_1 = [T_Foto36.max(), mean36_1, std_dev36_1]  # Estimativas iniciais para A, mu1 e sigma1
popt36_1, _ = curve_fit(gauss_sum, valores_selecionados_1, T_Foto36[(T_Piezo36 >= 4) & (T_Piezo36 <= 5)], p0=p036_1)

p036_2 = [T_Foto36[(T_Piezo36 >= 5) & (T_Piezo36 <= 6)].max(), mean36_2, std_dev36_2]  # Estimativas iniciais para B, mu2 e sigma2
popt36_2, _ = curve_fit(gauss, valores_selecionados_2, T_Foto36[(T_Piezo36 >= 5) & (T_Piezo36 <= 6)], p0=p036_2)

x_fit36 = np.linspace(4, 7)  # Intervalo para o ajuste
#x_fit36_2 = np.linspace(5, 6, 100)

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo36, T_Foto36, color='red')
plt.plot(x_fit36, gauss_sum(x_fit36, *popt36_1), color='blue', label='Distribuição Normal (Gaussiana) Ajustada')

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula36 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() 

### aula3_7.lab ###
'''
with open('aula3_7.lab', 'r') as file:
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
valores_selecionados = T_Piezo37[(T_Piezo37 >= 2) & (T_Piezo37 <= 4)]

# Calcular a média e o desvio padrão dos dados selecionados
mean37 = np.mean(valores_selecionados)
std_dev37 = np.std(valores_selecionados)

# Definir a função gaussiana
def gauss(x, A, mu, sigma):
    return (A * np.exp(-(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2)

# Ajustar a gaussiana apenas para o intervalo selecionado
# p037 = [T_Foto37.max(), mean37, std_dev37]  # Estimativas iniciais para A, mu e sigma
# popt37, _ = curve_fit(gauss, valores_selecionados, T_Foto37[(T_Piezo37 >= 3) & (T_Piezo37 <= 4)], p0=p037)

# x_fit37 = np.linspace(3, 4.5, 100)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo37, T_Foto37, color='black')

#plt.plot(x_fit37, gauss(x_fit37, *popt37), color='red', label='Distribuição Normal (Gaussiana) Ajustada')
plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula37 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.show() '''