from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# Definir a função gaussiana
def gauss(x, A, mu, sigma, d):
    return (A * np.exp(0.5* -(x - mu) ** 2 / (sigma ** 2)))/np.sqrt(2*np.pi*sigma**2) + d

# Definir a função gaussiana
def gauss_sum(x, A, mu1, sigma1, B, mu2, sigma2,d):
    return (A * np.exp(0.5* -(x - mu1) ** 2 / (sigma1 ** 2)))/np.sqrt(2*np.pi*sigma1**2) + (B * np.exp(0.5*-(x - mu2) ** 2 / (sigma2 ** 2)))/np.sqrt(2*np.pi*sigma2**2) + d

### aula4_1.lab ###

with open('aula4_1.lab', 'r') as file:
    lines = file.readlines()

data41 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data41.append(values)

df41 = pd.DataFrame(data41, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo41 = np.array(df41['T.Piezo'])
T_Foto41 = np.array(df41['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo41[(T_Piezo41 >= 0) & (T_Piezo41 <= 2)]
valores_selecionados_2 = T_Piezo41[(T_Piezo41 >= 4) & (T_Piezo41 <= 5.5)]
valores_selecionados_3 = T_Piezo41[(T_Piezo41 >= 7.5) & (T_Piezo41 <= 9)]

# Calcular a média e o desvio padrão dos dados selecionados
mean41_1 = np.mean(valores_selecionados_1)
std_dev41_1 = np.std(valores_selecionados_1)
mean41_2 = np.mean(valores_selecionados_2)
std_dev41_2 = np.std(valores_selecionados_2)
mean41_3 = np.mean(valores_selecionados_3)
std_dev41_3 = np.std(valores_selecionados_3)

# Ajustar a gaussiana apenas para o intervalo selecionado
p041_1 = [T_Foto41.max(), mean41_1, std_dev41_1, 5]  # Estimativas iniciais para A, mu e sigma
popt41_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto41[(T_Piezo41 >= 0) & (T_Piezo41 <= 2)], p0=p041_1)
A1_error, mu1_error, sigma1_error, d1_error = np.sqrt(np.diag(covariance1))

p041_2 = [T_Foto41.max(), mean41_2, std_dev41_2, 5]  # Estimativas iniciais para A, mu e sigma
popt41_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto41[(T_Piezo41 >= 4) & (T_Piezo41 <= 5.5)], p0=p041_2)
A2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

p041_3 = [T_Foto41.max(), mean41_3, std_dev41_3, 5]  # Estimativas iniciais para A, mu e sigma
popt41_3, covariance3 = curve_fit(gauss, valores_selecionados_3, T_Foto41[(T_Piezo41 >= 7.5) & (T_Piezo41 <= 9)], p0=p041_3)
A3_error, mu3_error, sigma3_error, d3_error = np.sqrt(np.diag(covariance3))

x_fit41_1 = np.linspace(0, 2)  # Intervalo para o ajuste
x_fit41_2 = np.linspace(4, 5.5)  # Intervalo para o ajuste
x_fit41_3 = np.linspace(7.5, 9)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo41, T_Foto41)

plt.plot(x_fit41_1, gauss(x_fit41_1, *popt41_1), label=r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt41_1[0], A1_error, popt41_1[1], mu1_error, popt41_1[2], sigma1_error, popt41_1[3], d1_error))
plt.plot(x_fit41_2, gauss(x_fit41_2, *popt41_2), label=r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt41_2[0], A2_error, popt41_2[1], mu2_error, popt41_2[2], sigma2_error, popt41_2[3], d2_error))
plt.plot(x_fit41_3, gauss(x_fit41_3, *popt41_3), label=r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt41_3[0], A3_error, popt41_3[1], mu3_error, popt41_3[2], sigma3_error, popt41_3[3], d3_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula41')

x_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.savefig('aula41.png')
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt41_1[0])
print('mu1 = ', popt41_1[1])
print('sigma1 = ', popt41_1[2])
print('A2 = ', popt41_2[0])
print('mu2 = ', popt41_2[1])
print('sigma2 = ', popt41_2[2])
print('A3 = ', popt41_3[0])
print('mu3 = ', popt41_3[1])
print('sigma3 = ', popt41_3[2])
print('d = ', popt41_3[3])
print('\n') 

### aula4_9.lab ###

with open('aula4_9.lab', 'r') as file:
    lines = file.readlines()

data49 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data49.append(values)

df49 = pd.DataFrame(data49, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo49 = np.array(df49['T.Piezo'])
T_Foto49 = np.array(df49['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo49[(T_Piezo49 >= 0.5) & (T_Piezo49 <= 3)]
valores_selecionados_2 = T_Piezo49[(T_Piezo49 >= 4) & (T_Piezo49 <= 6)]
valores_selecionados_3 = T_Piezo49[(T_Piezo49 >= 7.5) & (T_Piezo49 <= 10)]

# Calcular a média e o desvio padrão dos dados selecionados
mean49_1 = np.mean(valores_selecionados_1)
std_dev49_1 = np.std(valores_selecionados_1)
mean49_2 = np.mean(valores_selecionados_2)
std_dev49_2 = np.std(valores_selecionados_2)
mean49_3 = np.mean(valores_selecionados_3)
std_dev49_3 = np.std(valores_selecionados_3)

# Ajustar a gaussiana apenas para o intervalo selecionado
p049_1 = [T_Foto49.max(), mean49_1, std_dev49_1, 5]  # Estimativas iniciais para A, mu e sigma
popt49_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto49[(T_Piezo49 >= 0.5) & (T_Piezo49 <= 3)], p0=p049_1)
A1_error, mu1_error, sigma1_error, d1_error = np.sqrt(np.diag(covariance1))

p049_2 = [T_Foto49.max(), mean49_2, std_dev49_2, 5]  # Estimativas iniciais para A, mu e sigma
popt49_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto49[(T_Piezo49 >= 4) & (T_Piezo49 <= 6)], p0=p049_2)
A2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

p049_3 = [T_Foto49.max(), mean49_3, std_dev49_3, 5]  # Estimativas iniciais para A, mu e sigma
popt49_3, covariance3 = curve_fit(gauss, valores_selecionados_3, T_Foto49[(T_Piezo49 >= 7.5) & (T_Piezo49 <= 10)], p0=p049_3)
A3_error, mu3_error, sigma3_error, d3_error = np.sqrt(np.diag(covariance3))

x_fit49_1 = np.linspace(0.5, 3)  # Intervalo para o ajuste
x_fit49_2 = np.linspace(4, 6)  # Intervalo para o ajuste
x_fit49_3 = np.linspace(7.5, 10)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo49, T_Foto49)

plt.plot(x_fit49_1, gauss(x_fit49_1, *popt49_1), label=r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt49_1[0], A1_error, popt49_1[1], mu1_error, popt49_1[2], sigma1_error, popt49_1[3], d1_error))
plt.plot(x_fit49_2, gauss(x_fit49_2, *popt49_2), label=r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt49_2[0], A2_error, popt49_2[1], mu2_error, popt49_2[2], sigma2_error, popt49_2[3], d2_error))
plt.plot(x_fit49_3, gauss(x_fit49_3, *popt49_3), label=r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt49_3[0], A3_error, popt49_3[1], mu3_error, popt49_3[2], sigma3_error, popt49_3[3], d3_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula49')

x_values_to_display = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.savefig('aula49.png')
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt49_1[0])
print('mu1 = ', popt49_1[1])
print('sigma1 = ', popt49_1[2])
print('A2 = ', popt49_2[0])
print('mu2 = ', popt49_2[1])
print('sigma2 = ', popt49_2[2])
print('A3 = ', popt49_3[0])
print('mu3 = ', popt49_3[1])
print('sigma3 = ', popt49_3[2])
print('d = ', popt49_3[3])
print('\n') 

### aula4_10.lab ###

with open('aula4_10.lab', 'r') as file:
    lines = file.readlines()

data410 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data410.append(values)

df410 = pd.DataFrame(data410, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo410 = np.array(df410['T.Piezo'])
T_Foto410 = np.array(df410['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo410[(T_Piezo410 >= 0.5) & (T_Piezo410 <= 3)]
valores_selecionados_2 = T_Piezo410[(T_Piezo410 >= 4) & (T_Piezo410 <= 6.5)]
valores_selecionados_3 = T_Piezo410[(T_Piezo410 >= 7.5) & (T_Piezo410 <= 10)]

# Calcular a média e o desvio padrão dos dados selecionados
mean410_1 = np.mean(valores_selecionados_1)
std_dev410_1 = np.std(valores_selecionados_1)
mean410_2 = np.mean(valores_selecionados_2)
std_dev410_2 = np.std(valores_selecionados_2)
mean410_3 = np.mean(valores_selecionados_3)
std_dev410_3 = np.std(valores_selecionados_3)

# Ajustar a gaussiana apenas para o intervalo selecionado
p0410_1 = [T_Foto410.max(), mean410_1, std_dev410_1, 5]  # Estimativas iniciais para A, mu e sigma
popt410_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto410[(T_Piezo410 >= 0.5) & (T_Piezo410 <= 3)], p0=p0410_1)
A1_error, mu1_error, sigma1_error, d1_error = np.sqrt(np.diag(covariance1))

p0410_2 = [T_Foto410.max(), mean410_2, std_dev410_2, 5]  # Estimativas iniciais para A, mu e sigma
popt410_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto410[(T_Piezo410 >= 4) & (T_Piezo410 <= 6.5)], p0=p0410_2)
A2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

p0410_3 = [T_Foto410.max(), mean410_3, std_dev410_3, 5]  # Estimativas iniciais para A, mu e sigma
popt410_3, covariance3 = curve_fit(gauss, valores_selecionados_3, T_Foto410[(T_Piezo410 >= 7.5) & (T_Piezo410 <= 10)], p0=p0410_3)
A3_error, mu3_error, sigma3_error, d3_error = np.sqrt(np.diag(covariance3))

x_fit410_1 = np.linspace(0.5, 3)  # Intervalo para o ajuste
x_fit410_2 = np.linspace(4, 6.5)  # Intervalo para o ajuste
x_fit410_3 = np.linspace(7.5, 10)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo410, T_Foto410)

label1 = r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt410_1[0], A1_error, popt410_1[1], mu1_error, popt410_1[2], sigma1_error, popt410_1[3], d1_error)
plt.plot(x_fit410_1, gauss(x_fit410_1, *popt410_1), label=label1)

label2 = r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt410_2[0], A2_error, popt410_2[1], mu2_error, popt410_2[2], sigma2_error, popt410_2[3], d2_error)
plt.plot(x_fit410_2, gauss(x_fit410_2, *popt410_2), label=label2)

label3 = r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(popt410_3[0], A3_error, popt410_3[1], mu3_error, popt410_3[2], sigma3_error, popt410_3[3], d3_error)
plt.plot(x_fit410_3, gauss(x_fit410_3, *popt410_3), label=label3)

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula410')

x_values_to_display = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.savefig('aula410.png')
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt410_1[0])
print('mu1 = ', popt410_1[1])
print('sigma1 = ', popt410_1[2])
print('A2 = ', popt410_2[0])
print('mu2 = ', popt410_2[1])
print('sigma2 = ', popt410_2[2])
print('A3 = ', popt410_3[0])
print('mu3 = ', popt410_3[1])
print('sigma3 = ', popt410_3[2])
print('d = ', popt410_3[3])
print('\n') 

### aula4_11.lab ###

with open('aula4_11.lab', 'r') as file:
    lines = file.readlines()

data411 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data411.append(values)

df411 = pd.DataFrame(data411, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo411 = np.array(df411['T.Piezo'])
T_Foto411 = np.array(df411['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo411[(T_Piezo411 >= 0.5) & (T_Piezo411 <= 3)]
valores_selecionados_2 = T_Piezo411[(T_Piezo411 >= 4) & (T_Piezo411 <= 6.5)]
valores_selecionados_3 = T_Piezo411[(T_Piezo411 >= 7.5) & (T_Piezo411 <= 10)]

# Calcular a média e o desvio padrão dos dados selecionados
mean411_1 = np.mean(valores_selecionados_1)
std_dev411_1 = np.std(valores_selecionados_1)
mean411_2 = np.mean(valores_selecionados_2)
std_dev411_2 = np.std(valores_selecionados_2)
mean411_3 = np.mean(valores_selecionados_3)
std_dev411_3 = np.std(valores_selecionados_3)

# Ajustar a gaussiana apenas para o intervalo selecionado
p0411_1 = [T_Foto411.max(), mean411_1, std_dev411_1, 5]  # Estimativas iniciais para A, mu e sigma
popt411_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto411[(T_Piezo411 >= 0.5) & (T_Piezo411 <= 3)], p0=p0411_1)
A1_error, mu1_error, sigma1_error, d1_error = np.sqrt(np.diag(covariance1))

p0411_2 = [T_Foto411.max(), mean411_2, std_dev411_2, 5]  # Estimativas iniciais para A, mu e sigma
popt411_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto411[(T_Piezo411 >= 4) & (T_Piezo411 <= 6.5)], p0=p0411_2)
A2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

p0411_3 = [T_Foto411.max(), mean411_3, std_dev411_3, 5]  # Estimativas iniciais para A, mu e sigma
popt411_3, covariance3 = curve_fit(gauss, valores_selecionados_3, T_Foto411[(T_Piezo411 >= 7.5) & (T_Piezo411 <= 10)], p0=p0411_3)
A3_error, mu3_error, sigma3_error, d3_error = np.sqrt(np.diag(covariance3))

x_fit411_1 = np.linspace(0.5, 3)  # Intervalo para o ajuste
x_fit411_2 = np.linspace(4, 6.5)  # Intervalo para o ajuste
x_fit411_3 = np.linspace(7.5, 10)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo411, T_Foto411)

label1 = r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' % (popt411_1[0], A1_error, popt411_1[1], mu1_error, popt411_1[2], sigma1_error, popt411_1[3], d1_error)
label2 = r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' % (popt411_2[0], A2_error, popt411_2[1], mu2_error, popt411_2[2], sigma2_error, popt411_2[3], d2_error)
label3 = r'$A = %.3f \pm %.3f; \mu = %.3f \pm %.3f; \sigma = %.3f \pm %.3f; d = %.3f \pm %.3f$' % (popt411_3[0], A3_error, popt411_3[1], mu3_error, popt411_3[2], sigma3_error, popt411_3[3], d3_error)

plt.plot(x_fit411_1, gauss(x_fit411_1, *popt411_1), label=label1)
plt.plot(x_fit411_2, gauss(x_fit411_2, *popt411_2), label=label2)
plt.plot(x_fit411_3, gauss(x_fit411_3, *popt411_3), label=label3)

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula411')

x_values_to_display = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.savefig('aula411.png')
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt411_1[0])
print('mu1 = ', popt411_1[1])
print('sigma1 = ', popt411_1[2])
print('A2 = ', popt411_2[0])
print('mu2 = ', popt411_2[1])
print('sigma2 = ', popt411_2[2])
print('A3 = ', popt411_3[0])
print('mu3 = ', popt411_3[1])
print('sigma3 = ', popt411_3[2])
print('d = ', popt411_3[3])
print('\n') 

### aula4_12.lab ###

with open('aula4_12.lab', 'r') as file:
    lines = file.readlines()

data412 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data412.append(values)

df412 = pd.DataFrame(data412, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo412 = np.array(df412['T.Piezo'])
T_Foto412 = np.array(df412['T.Foto'])

# Selecionar o intervalo de T_Piezo entre 3 e 4
valores_selecionados_1 = T_Piezo412[(T_Piezo412 >= 1.5) & (T_Piezo412 <= 3)]
valores_selecionados_2 = T_Piezo412[(T_Piezo412 >= 5) & (T_Piezo412 <= 6.5)]
valores_selecionados_3 = T_Piezo412[(T_Piezo412 >= 8.5) & (T_Piezo412 <= 10)]

# Calcular a média e o desvio padrão dos dados selecionados
mean412_1 = np.mean(valores_selecionados_1)
std_dev412_1 = np.std(valores_selecionados_1)
mean412_2 = np.mean(valores_selecionados_2)
std_dev412_2 = np.std(valores_selecionados_2)
mean412_3 = np.mean(valores_selecionados_3)
std_dev412_3 = np.std(valores_selecionados_3)

# Ajustar a gaussiana apenas para o intervalo selecionado
p0412_1 = [T_Foto412.max(), mean412_1, std_dev412_1, 5]  # Estimativas iniciais para A, mu e sigma
popt412_1, covariance1 = curve_fit(gauss, valores_selecionados_1, T_Foto412[(T_Piezo412 >= 1.5) & (T_Piezo412 <= 3)], p0=p0412_1)
A1_error, mu1_error, sigma1_error, d1_error = np.sqrt(np.diag(covariance1))

p0412_2 = [T_Foto412.max(), mean412_2, std_dev412_2, 5]  # Estimativas iniciais para A, mu e sigma
popt412_2, covariance2 = curve_fit(gauss, valores_selecionados_2, T_Foto412[(T_Piezo412 >= 5) & (T_Piezo412 <= 6.5)], p0=p0412_2)
A2_error, mu2_error, sigma2_error, d2_error = np.sqrt(np.diag(covariance2))

p0412_3 = [T_Foto412.max(), mean412_3, std_dev412_3, 5]  # Estimativas iniciais para A, mu e sigma
popt412_3, covariance3 = curve_fit(gauss, valores_selecionados_3, T_Foto412[(T_Piezo412 >= 8.5) & (T_Piezo412 <= 10)], p0=p0412_3)
A3_error, mu3_error, sigma3_error, d3_error = np.sqrt(np.diag(covariance3))

x_fit412_1 = np.linspace(1.5, 3)  # Intervalo para o ajuste
x_fit412_2 = np.linspace(5, 6.5)  # Intervalo para o ajuste
x_fit412_3 = np.linspace(8.5, 10)  # Intervalo para o ajuste

# Plotagem dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))
plt.plot(T_Piezo412, T_Foto412)

plt.plot(x_fit412_1, gauss(x_fit412_1, *popt412_1), label=r'$A_1 = %.3f \pm %.3f; \mu_1 = %.3f \pm %.3f; \sigma_1 = %.3f \pm %.3f; d_1 = %.3f \pm %.3f$' %(popt412_1[0], A1_error, popt412_1[1], mu1_error, popt412_1[2], sigma1_error, popt412_1[3], d1_error))
plt.plot(x_fit412_2, gauss(x_fit412_2, *popt412_2), label=r'$A_2 = %.3f \pm %.3f; \mu_2 = %.3f \pm %.3f; \sigma_2 = %.3f \pm %.3f; d_2 = %.3f \pm %.3f$' %(popt412_2[0], A2_error, popt412_2[1], mu2_error, popt412_2[2], sigma2_error, popt412_2[3], d2_error))
plt.plot(x_fit412_3, gauss(x_fit412_3, *popt412_3), label=r'$A_3 = %.3f \pm %.3f; \mu_3 = %.3f \pm %.3f; \sigma_3 = %.3f \pm %.3f; d_3 = %.3f \pm %.3f$' %(popt412_3[0], A3_error, popt412_3[1], mu3_error, popt412_3[2], sigma3_error, popt412_3[3], d3_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula412')

x_values_to_display = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
y_values_to_display = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend()
plt.savefig('aula412.png')
plt.show() 

print('\n')
print('Gaussian parameters: ')
print('A1 = ', popt412_1[0])
print('mu1 = ', popt412_1[1])
print('sigma1 = ', popt412_1[2])
print('A2 = ', popt412_2[0])
print('mu2 = ', popt412_2[1])
print('sigma2 = ', popt412_2[2])
print('A3 = ', popt412_3[0])
print('mu3 = ', popt412_3[1])
print('sigma3 = ', popt412_3[2])
print('d = ', popt412_3[3])
print('\n') 

### aula4_2.lab ###

with open('aula4_2.lab', 'r') as file:
    lines = file.readlines()

data42 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data42.append(values)

df42 = pd.DataFrame(data42, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo42 = np.array(df42['T.Piezo'])
T_Foto42 = np.array(df42['T.Foto'])

valores_selecionados_3 = T_Piezo42[(T_Piezo42 >= 4.) & (T_Piezo42 <= 5)]
valores_selecionados_4 = T_Piezo42[(T_Piezo42 >= 5.5) & (T_Piezo42 <= 6)]
valores_selecionados_1 = T_Piezo42[(T_Piezo42 >= 0.75) & (T_Piezo42 <= 1.5)]
valores_selecionados_2 = T_Piezo42[(T_Piezo42 >= 1.5) & (T_Piezo42 <= 3)]
valores_selecionados_5 = T_Piezo42[(T_Piezo42 >= 8) & (T_Piezo42 <= 8.75)]
valores_selecionados_6 = T_Piezo42[(T_Piezo42 >= 8.75) & (T_Piezo42 <= 9.5)]

# Calcular a média e o desvio padrão dos dados selecionados
mean42_1 = np.mean(valores_selecionados_1)
mean42_2 = np.mean(valores_selecionados_2)
std_dev42_1 = np.std(valores_selecionados_1) 
std_dev42_2 = np.std(valores_selecionados_2)
mean42_3 = np.mean(valores_selecionados_3)
std_dev42_3 = np.std(valores_selecionados_3) 
mean42_4 = np.mean(valores_selecionados_4)
std_dev42_4 = np.std(valores_selecionados_4)
mean42_5 = np.mean(valores_selecionados_5)
std_dev42_5 = np.std(valores_selecionados_5) 
mean42_6 = np.mean(valores_selecionados_6)
std_dev42_6 = np.std(valores_selecionados_6) 

# Ajustar a gaussiana apenas para o intervalo selecionado
p042_1 = [T_Foto42.max(), mean42_1, std_dev42_1,T_Foto42[(T_Piezo42 >= 1.5) & (T_Piezo42 <= 3)].max(), mean42_2, std_dev42_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt42_1, covariance1 = curve_fit(gauss_sum, T_Piezo42, T_Foto42, p0=p042_1)
x_fit42_1 = np.linspace(0.5, 3)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt42_1[0] , popt42_1[1], popt42_1[2] , popt42_1[3], popt42_1[4], popt42_1[5], popt42_1[6]
A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

p042_2 = [T_Foto42.max(), mean42_3, std_dev42_3,T_Foto42[(T_Piezo42 >= 5.5) & (T_Piezo42 <= 6)].max(), mean42_4, std_dev42_4, 0.5]  # Estimativas iniciais para A, mu1 e sigma1
popt42_2, covariance2 = curve_fit(gauss_sum, T_Piezo42, T_Foto42, p0=p042_2)
x_fit42_2 = np.linspace(4.5, 6)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt42_2[0] , popt42_2[1], popt42_2[2] , popt42_2[3], popt42_2[4], popt42_2[5], popt42_2[6]
A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

p042_3 = [T_Foto42.max(), mean42_5, std_dev42_5, T_Foto42[(T_Piezo42 >= 8.75) & (T_Piezo42 <= 9.5)].max(), mean42_6, std_dev42_6, 5]  # Estimativas iniciais para A, mu e sigma
popt42_3, covariance3 = curve_fit(gauss_sum, T_Piezo42, T_Foto42, p0=p042_3)
x_fit42_3 = np.linspace(8, 9.5)  # Intervalo para o ajuste
A5 , mu5 , sigma5, A6, mu6, sigma6, d = popt42_3[0] , popt42_3[1], popt42_3[2] , popt42_3[3], popt42_3[4], popt42_3[5], popt42_3[6]
A5_error, mu5_error, sigma5_error, A6_error, mu6_error, sigma6_error, d3_error = np.sqrt(np.diag(covariance3))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo42, T_Foto42)

plt.plot(x_fit42_1, gauss_sum(x_fit42_1, *popt42_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error))
plt.plot(x_fit42_2, gauss_sum(x_fit42_2, *popt42_2), label=r'$A3 = %.3f \pm %.3f; \mu 3 = %.3f \pm %.3f; \sigma 3 = %.3f \pm %.3f; A4 = %.3f \pm %.3f; \mu 4 = %.3f \pm %.3f; \sigma 4 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error))
plt.plot(x_fit42_3, gauss_sum(x_fit42_3, *popt42_3), label=r'$A5 = %.3f \pm %.3f; \mu 5 = %.3f \pm %.3f; \sigma 5 = %.3f \pm %.3f; A6 = %.3f \pm %.3f; \mu 6 = %.3f \pm %.3f; \sigma 6 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A5, A5_error, mu5, mu5_error, sigma5, sigma5_error, A6, A6_error, mu6, mu6_error, sigma6, sigma6_error, d, d3_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula42 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('aula42.png')
plt.show() 

### aula4_3.lab ###

with open('aula4_3.lab', 'r') as file:
    lines = file.readlines()

data43 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data43.append(values)

df43 = pd.DataFrame(data43, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo43 = np.array(df43['T.Piezo'])
T_Foto43 = np.array(df43['T.Foto'])

valores_selecionados_3 = T_Piezo43[(T_Piezo43 >= 4.5) & (T_Piezo43 <= 5.5)]
valores_selecionados_4 = T_Piezo43[(T_Piezo43 >= 5.5) & (T_Piezo43 <= 6.5)]
valores_selecionados_1 = T_Piezo43[(T_Piezo43 >= 1) & (T_Piezo43 <= 2)]
valores_selecionados_2 = T_Piezo43[(T_Piezo43 >= 2) & (T_Piezo43 <= 3)]
valores_selecionados_5 = T_Piezo43[(T_Piezo43 >= 7.75) & (T_Piezo43 <= 9)]
valores_selecionados_6 = T_Piezo43[(T_Piezo43 >= 9.2) & (T_Piezo43 <= 10)]

# Calcular a média e o desvio padrão dos dados selecionados
mean43_1 = np.mean(valores_selecionados_1)
mean43_2 = np.mean(valores_selecionados_2)
std_dev43_1 = np.std(valores_selecionados_1) 
std_dev43_2 = np.std(valores_selecionados_2)
mean43_3 = np.mean(valores_selecionados_3)
std_dev43_3 = np.std(valores_selecionados_3) 
mean43_4 = np.mean(valores_selecionados_4)
std_dev43_4 = np.std(valores_selecionados_4)
mean43_5 = np.mean(valores_selecionados_5)
std_dev43_5 = np.std(valores_selecionados_5) 
mean43_6 = np.mean(valores_selecionados_6)
std_dev43_6 = np.std(valores_selecionados_6) 

# Ajustar a gaussiana apenas para o intervalo selecionado
p043_1 = [T_Foto43.max(), mean43_1, std_dev43_1,T_Foto43[(T_Piezo43 >= 2) & (T_Piezo43 <= 3)].max(), mean43_2, std_dev43_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt43_1, covariance1 = curve_fit(gauss_sum, T_Piezo43, T_Foto43, p0=p043_1)
x_fit43_1 = np.linspace(1, 3)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt43_1[0] , popt43_1[1], popt43_1[2] , popt43_1[3], popt43_1[4], popt43_1[5], popt43_1[6]
A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

p043_2 = [T_Foto43.max(), mean43_3, std_dev43_3,T_Foto43[(T_Piezo43>= 5.5) & (T_Piezo43 <= 6.5)].max(), mean43_4, std_dev43_4, 0.5]  # Estimativas iniciais para A, mu1 e sigma1
popt43_2, covariance2 = curve_fit(gauss_sum, T_Piezo43, T_Foto43, p0=p043_2)
x_fit43_2 = np.linspace(4.5, 6.5)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt43_2[0] , popt43_2[1], popt43_2[2] , popt43_2[3], popt43_2[4], popt43_2[5], popt43_2[6]
A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

p043_3 = [T_Foto43.max(), mean43_5, std_dev43_5, T_Foto43[(T_Piezo43>= 9.2) & (T_Piezo43 <= 10)].max(), mean43_6, std_dev43_6, 5]  # Estimativas iniciais para A, mu e sigma
popt43_3, covariance3 = curve_fit(gauss_sum, T_Piezo43, T_Foto43, p0=p043_3)
x_fit43_3 = np.linspace(7.75, 10)  # Intervalo para o ajuste
A5 , mu5 , sigma5, A6, mu6, sigma6, d = popt43_3[0] , popt43_3[1], popt43_3[2] , popt43_3[3], popt43_3[4], popt43_3[5], popt43_3[6]
A5_error, mu5_error, sigma5_error, A6_error, mu6_error, sigma6_error, d3_error = np.sqrt(np.diag(covariance3))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo43, T_Foto43)

# Plot da primeira gaussiana ajustada
label1 = r'$A1 = %.3f \pm %.3f$; $\mu 1 = %.3f \pm %.3f$; $\sigma 1 = %.3f \pm %.3f$; $A2 = %.3f \pm %.3f$; $\mu 2 = %.3f \pm %.3f$; $\sigma 2  = %.3f \pm %.3f$; $d = %.3f \pm %.3f$' % (A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error)
plt.plot(x_fit43_1, gauss_sum(x_fit43_1, *popt43_1), label=label1)

# Plot da segunda gaussiana ajustada
label2 = r'$A3 = %.3f \pm %.3f$; $\mu 3 = %.3f \pm %.3f$; $\sigma 3 = %.3f \pm %.3f$; $A4 = %.3f \pm %.3f$; $\mu 4 = %.3f \pm %.3f$; $\sigma 4 = %.3f \pm %.3f$; $d = %.3f \pm %.3f$' % (A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error)
plt.plot(x_fit43_2, gauss_sum(x_fit43_2, *popt43_2), label=label2)

# Plot da terceira gaussiana ajustada
label3 = r'$A5 = %.3f \pm %.3f$; $\mu 5 = %.3f \pm %.3f$; $\sigma 5 = %.3f \pm %.3f$; $A6 = %.3f \pm %.3f$; $\mu 6 = %.3f \pm %.3f$; $\sigma 6 = %.3f \pm %.3f$; $d = %.3f \pm %.3f$' % (A5, A5_error, mu5, mu5_error, sigma5, sigma5_error, A6, A6_error, mu6, mu6_error, sigma6, sigma6_error, d, d3_error)
plt.plot(x_fit43_3, gauss_sum(x_fit43_3, *popt43_3), label=label3)

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula43 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('aula43.png')
plt.show() 

### aula4_4.lab ###

with open('aula4_4.lab', 'r') as file:
    lines = file.readlines()

data44 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data44.append(values)

df44 = pd.DataFrame(data44, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo44 = np.array(df44['T.Piezo'])
T_Foto44 = np.array(df44['T.Foto'])

valores_selecionados_3 = T_Piezo44[(T_Piezo44 >= 5) & (T_Piezo44 <= 6)]
valores_selecionados_4 = T_Piezo44[(T_Piezo44 >= 6) & (T_Piezo44 <= 7)]
valores_selecionados_1 = T_Piezo44[(T_Piezo44 >= 1.5) & (T_Piezo44 <= 2.25)]
valores_selecionados_2 = T_Piezo44[(T_Piezo44 >= 2.25) & (T_Piezo44 <= 3.5)]
valores_selecionados_5 = T_Piezo44[(T_Piezo44 >= 8.75) & (T_Piezo44 <= 9.5)]
valores_selecionados_6 = T_Piezo44[(T_Piezo44 >= 9.5) & (T_Piezo44 <= 10)]

# Calcular a média e o desvio padrão dos dados selecionados
mean44_1 = np.mean(valores_selecionados_1)
mean44_2 = np.mean(valores_selecionados_2)
std_dev44_1 = np.std(valores_selecionados_1) 
std_dev44_2 = np.std(valores_selecionados_2)
mean44_3 = np.mean(valores_selecionados_3)
std_dev44_3 = np.std(valores_selecionados_3) 
mean44_4 = np.mean(valores_selecionados_4)
std_dev44_4 = np.std(valores_selecionados_4)
mean44_5 = np.mean(valores_selecionados_5)
std_dev44_5 = np.std(valores_selecionados_5) 
mean44_6 = np.mean(valores_selecionados_6)
std_dev44_6 = np.std(valores_selecionados_6) 

# Ajustar a gaussiana apenas para o intervalo selecionado
p044_1 = [T_Foto44.max(), mean44_1, std_dev44_1,T_Foto44[(T_Piezo44 >= 2.25) & (T_Piezo44 <= 3.5)].max(), mean44_2, std_dev44_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt44_1, covariance1 = curve_fit(gauss_sum, T_Piezo44, T_Foto44, p0=p044_1)
x_fit44_1 = np.linspace(1.5, 3.5)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt44_1[0] , popt44_1[1], popt44_1[2] , popt44_1[3], popt44_1[4], popt44_1[5], popt44_1[6]
A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

p044_2 = [T_Foto44.max(), mean44_3, std_dev44_3,T_Foto44[(T_Piezo44>= 6) & (T_Piezo44 <= 7)].max(), mean44_4, std_dev44_4, 0.5]  # Estimativas iniciais para A, mu1 e sigma1
popt44_2, covariance2 = curve_fit(gauss_sum, T_Piezo44, T_Foto44, p0=p044_2)
x_fit44_2 = np.linspace(5, 7)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt44_2[0] , popt44_2[1], popt44_2[2] , popt44_2[3], popt44_2[4], popt44_2[5], popt44_2[6]
A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

p044_3 = [T_Foto44.max(), mean44_5, std_dev44_5, T_Foto44[(T_Piezo44>= 9.5) & (T_Piezo44 <= 10)].max(), mean44_6, std_dev44_6, 5]  # Estimativas iniciais para A, mu e sigma
popt44_3, covariance3 = curve_fit(gauss_sum, T_Piezo44, T_Foto44, p0=p044_3)
x_fit44_3 = np.linspace(8.75, 10)  # Intervalo para o ajuste
A5 , mu5 , sigma5, A6, mu6, sigma6, d = popt44_3[0] , popt44_3[1], popt44_3[2] , popt44_3[3], popt44_3[4], popt44_3[5], popt44_3[6]
A5_error, mu5_error, sigma5_error, A6_error, mu6_error, sigma6_error, d3_error = np.sqrt(np.diag(covariance3))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo44, T_Foto44)
#plt.plot(, label=r'$%3.f \cdot \frac{1}{\sqrt{2\pi%3.f}}e^{\frac{-(x-%.3f)^2}{%.3f^2}}$,  $\mu_2$ = %.3f, $\sigma_2$ = %.3f, d = %.3f' % (A1, sigma1, mu1, sigma1, mu2, sigma2, d))

label1 = r'$A1 = %.3f \pm %.3f$; $\mu 1 = %.3f \pm %.3f$; $\sigma 1 = %.3f \pm %.3f$; $A2 = %.3f \pm %.3f$; $\mu 2 = %.3f \pm %.3f$; $\sigma 2 = %.3f \pm %.3f$; $d = %.3f \pm %.3f$' % (A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error)
plt.plot(x_fit44_1, gauss_sum(x_fit44_1, *popt44_1), label=label1)

label2 = r'$A3 = %.3f \pm %.3f$; $\mu 3 = %.3f \pm %.3f$; $\sigma 3 = %.3f \pm %.3f$; $A4 = %.3f \pm %.3f$; $\mu 4 = %.3f \pm %.3f$; $\sigma 4 = %.3f \pm %.3f$; $d = %.3f \pm %.3f$' % (A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error)
plt.plot(x_fit44_2, gauss_sum(x_fit44_2, *popt44_2), label=label2)

label3 = r'$A5 = %.3f \pm %.3f$; $\mu 5 = %.3f \pm %.3f$; $\sigma 5 = %.3f \pm %.3f$; $A6 = %.3f \pm %.3f$; $\mu 6 = %.3f \pm %.3f$; $\sigma 6 = %.3f \pm %.3f$; $d = %.3f \pm %.3f$' % (A5, A5_error, mu5, mu5_error, sigma5, sigma5_error, A6, A6_error, mu6, mu6_error, sigma6, sigma6_error, d, d3_error)
plt.plot(x_fit44_3, gauss_sum(x_fit44_3, *popt44_3), label=label3)

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula44 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('aula44.png')
plt.show() 

### aula4_7.lab ###

with open('aula4_7.lab', 'r') as file:
    lines = file.readlines()

data47 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data47.append(values)

df47 = pd.DataFrame(data47, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo47 = np.array(df47['T.Piezo'])
T_Foto47 = np.array(df47['T.Foto'])

valores_selecionados_3 = T_Piezo47[(T_Piezo47 >= 4.5) & (T_Piezo47 <= 5)]
valores_selecionados_4 = T_Piezo47[(T_Piezo47 >= 5.5) & (T_Piezo47 <= 6)]
valores_selecionados_1 = T_Piezo47[(T_Piezo47 >= 0.5) & (T_Piezo47 <= 1.5)]
valores_selecionados_2 = T_Piezo47[(T_Piezo47 >= 1.5) & (T_Piezo47 <= 2.5)]
valores_selecionados_5 = T_Piezo47[(T_Piezo47 >= 8.25) & (T_Piezo47 <= 8.4)]
valores_selecionados_6 = T_Piezo47[(T_Piezo47 >= 8.75) & (T_Piezo47 <= 9.25)]

# Calcular a média e o desvio padrão dos dados selecionados
mean47_1 = np.mean(valores_selecionados_1)
mean47_2 = np.mean(valores_selecionados_2)
std_dev47_1 = np.std(valores_selecionados_1) 
std_dev47_2 = np.std(valores_selecionados_2)
mean47_3 = np.mean(valores_selecionados_3)
std_dev47_3 = np.std(valores_selecionados_3) 
mean47_4 = np.mean(valores_selecionados_4)
std_dev47_4 = np.std(valores_selecionados_4)
mean47_5 = np.mean(valores_selecionados_5)
std_dev47_5 = np.std(valores_selecionados_5) 
mean47_6 = np.mean(valores_selecionados_6)
std_dev47_6 = np.std(valores_selecionados_6) 

# Ajustar a gaussiana apenas para o intervalo selecionado
p047_1 = [T_Foto47.max(), mean47_1, std_dev47_1,T_Foto47[(T_Piezo47 >= 1.5) & (T_Piezo47 <= 2.5)].max(), mean47_2, std_dev47_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt47_1, covariance1 = curve_fit(gauss_sum, T_Piezo47, T_Foto47, p0=p047_1)
x_fit47_1 = np.linspace(0.5, 2.5)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt47_1[0] , popt47_1[1], popt47_1[2] , popt47_1[3], popt47_1[4], popt47_1[5], popt47_1[6]
A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

p047_2 = [T_Foto47.max(), mean47_3, std_dev47_3,T_Foto47[(T_Piezo47>= 5.5) & (T_Piezo47 <= 6)].max(), mean47_4, std_dev47_4, 0.5]  # Estimativas iniciais para A, mu1 e sigma1
popt47_2, covariance2 = curve_fit(gauss_sum, T_Piezo47, T_Foto47, p0=p047_2)
x_fit47_2 = np.linspace(4.5, 6)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt47_2[0] , popt47_2[1], popt47_2[2] , popt47_2[3], popt47_2[4], popt47_2[5], popt47_2[6]
A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

p047_3 = [T_Foto47.max(), mean47_5, std_dev47_5, T_Foto47[(T_Piezo47>= 8.75) & (T_Piezo47 <= 9.25)].max(), mean47_6, std_dev47_6, 5]  # Estimativas iniciais para A, mu e sigma
popt47_3, covariance3 = curve_fit(gauss_sum, T_Piezo47, T_Foto47, p0=p047_3)
x_fit47_3 = np.linspace(8.25, 9.25)  # Intervalo para o ajuste
A5 , mu5 , sigma5, A6, mu6, sigma6, d = popt47_3[0] , popt47_3[1], popt47_3[2] , popt47_3[3], popt47_3[4], popt47_3[5], popt47_3[6]
A5_error, mu5_error, sigma5_error, A6_error, mu6_error, sigma6_error, d3_error = np.sqrt(np.diag(covariance3))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo47, T_Foto47)
#plt.plot(, label=r'$%3.f \cdot \frac{1}{\sqrt{2\pi%3.f}}e^{\frac{-(x-%.3f)^2}{%.3f^2}}$,  $\mu_2$ = %.3f, $\sigma_2$ = %.3f, d = %.3f' % (A1, sigma1, mu1, sigma1, mu2, sigma2, d))

label3 = r'$A5 = %.3f \pm %.3f$; $\mu 5 = %.3f \pm %.3f$; $\sigma 5 = %.3f \pm %.3f$; $A6 = %.3f \pm %.3f$; $\mu 6 = %.3f \pm %.3f$; $\sigma 6 = %.3f \pm %.3f$; $d = %.3f \pm %.3f$' % (A5, A5_error, mu5, mu5_error, sigma5, sigma5_error, A6, A6_error, mu6, mu6_error, sigma6, sigma6_error, d, d3_error)

plt.plot(x_fit47_1, gauss_sum(x_fit47_1, *popt47_1), label=r'$A1 = %.3f \pm %.3f$; $\mu 1 = %.3f \pm %.3f$; $\sigma 2 = %.3f \pm %.3f$; $A2 = %.3f \pm %.3f$; $\mu 2 = %.3f \pm %.3f$; $\sigma 2  = %.3f \pm %.3f$; $d = %.3f \pm %.3f$' %(A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error))
plt.plot(x_fit47_2, gauss_sum(x_fit47_2, *popt47_2), label=r'$A3 = %.3f \pm %.3f$; $\mu 3 = %.3f \pm %.3f$; $\sigma 3 = %.3f \pm %.3f$; $A4 = %.3f \pm %.3f$; $\mu 4 = %.3f \pm %.3f$; $\sigma 4 = %.3f \pm %.3f$; $d = %.3f \pm %.3f$' %(A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error))
plt.plot(x_fit47_3, gauss_sum(x_fit47_3, *popt47_3), label=label3)

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula47 ')


x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('aula47.png')
plt.show() 

### aula4_8.lab ###

with open('aula4_8.lab', 'r') as file:
    lines = file.readlines()

data48 = []
for line in lines:
    values = line.strip().split()
    values = [float(v) for v in values]
    data48.append(values)

df48 = pd.DataFrame(data48, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])

T_Piezo48 = np.array(df48['T.Piezo'])
T_Foto48 = np.array(df48['T.Foto'])

valores_selecionados_3 = T_Piezo48[(T_Piezo48 >= 5) & (T_Piezo48 <= 6)]
valores_selecionados_4 = T_Piezo48[(T_Piezo48 >= 6) & (T_Piezo48 <= 6.5)]
valores_selecionados_1 = T_Piezo48[(T_Piezo48 >= 1.5) & (T_Piezo48 <= 1.75)]
valores_selecionados_2 = T_Piezo48[(T_Piezo48 >= 1.5) & (T_Piezo48 <= 2.5)]
valores_selecionados_5 = T_Piezo48[(T_Piezo48 >= 8.75) & (T_Piezo48 <= 9.25)]
valores_selecionados_6 = T_Piezo48[(T_Piezo48 >= 9.4) & (T_Piezo48 <= 9.6)]

# Calcular a média e o desvio padrão dos dados selecionados
mean48_1 = np.mean(valores_selecionados_1)
mean48_2 = np.mean(valores_selecionados_2)
std_dev48_1 = np.std(valores_selecionados_1) 
std_dev48_2 = np.std(valores_selecionados_2)
mean48_3 = np.mean(valores_selecionados_3)
std_dev48_3 = np.std(valores_selecionados_3) 
mean48_4 = np.mean(valores_selecionados_4)
std_dev48_4 = np.std(valores_selecionados_4)
mean48_5 = np.mean(valores_selecionados_5)
std_dev48_5 = np.std(valores_selecionados_5) 
mean48_6 = np.mean(valores_selecionados_6)
std_dev48_6 = np.std(valores_selecionados_6) 

# Ajustar a gaussiana apenas para o intervalo selecionado
p048_1 = [T_Foto48.max(), mean48_1, std_dev48_1,T_Foto48[(T_Piezo48 >= 1.5) & (T_Piezo48 <= 2.5)].max(), mean48_2, std_dev48_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt48_1, covariance1 = curve_fit(gauss_sum, T_Piezo48, T_Foto48, p0=p048_1)
x_fit48_1 = np.linspace(0.5, 3)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt48_1[0] , popt48_1[1], popt48_1[2] , popt48_1[3], popt48_1[4], popt48_1[5], popt48_1[6]
A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

p048_2 = [T_Foto48.max(), mean48_3, 0.1,T_Foto48[(T_Piezo48>= 6) & (T_Piezo48 <= 7)].max(), mean48_4, std_dev48_4, 0.5]  # Estimativas iniciais para A, mu1 e sigma1
popt48_2, covariance2 = curve_fit(gauss_sum, T_Piezo48, T_Foto48, p0=p048_2)
x_fit48_2 = np.linspace(4.5, 7)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt48_2[0] , popt48_2[1], popt48_2[2] , popt48_2[3], popt48_2[4], popt48_2[5], popt48_2[6]
A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

p048_3 = [1000*T_Foto48.max(), mean48_5, 0.01, T_Foto48[(T_Piezo48>= 9.4) & (T_Piezo48 <= 9.6)].max(), mean48_6, std_dev48_6, 5]  # Estimativas iniciais para A, mu e sigma
popt48_3, covariance3 = curve_fit(gauss_sum, T_Piezo48, T_Foto48, p0=p048_3)
x_fit48_3 = np.linspace(8.25, 9.75)  # Intervalo para o ajuste
A5 , mu5 , sigma5, A6, mu6, sigma6, d = popt48_3[0] , popt48_3[1], popt48_3[2] , popt48_3[3], popt48_3[4], popt48_3[5], popt48_3[6]
A5_error, mu5_error, sigma5_error, A6_error, mu6_error, sigma6_error, d3_error = np.sqrt(np.diag(covariance3))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo48, T_Foto48)
#plt.plot(, label=r'$%3.f \cdot \frac{1}{\sqrt{2\pi%3.f}}e^{\frac{-(x-%.3f)^2}{%.3f^2}}$,  $\mu_2$ = %.3f, $\sigma_2$ = %.3f, d = %.3f' % (A1, sigma1, mu1, sigma1, mu2, sigma2, d))

plt.plot(x_fit48_1, gauss_sum(x_fit48_1, *popt48_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2  = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error))
plt.plot(x_fit48_2, gauss_sum(x_fit48_2, *popt48_2), label=r'$A3 = %.3f \pm %.3f; \mu 3 = %.3f \pm %.3f; \sigma 3 = %.3f \pm %.3f; A4 = %.3f \pm %.3f; \mu 4 = %.3f \pm %.3f; \sigma 4 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error))
plt.plot(x_fit48_3, gauss_sum(x_fit48_3, *popt48_3), label=r'$A5 = %.3f \pm %.3f; \mu 5 = %.3f \pm %.3f; \sigma 5 = %.3f \pm %.3f; A6 = %.3f \pm %.3f; \mu 6 = %.3f \pm %.3f; \sigma 6 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A5, A5_error, mu5, mu5_error, sigma5, sigma5_error, A6, A6_error, mu6, mu6_error, sigma6, sigma6_error, d, d3_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('aula48 ')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('aula48.png')
plt.show() 