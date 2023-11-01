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
def gauss_sum(x, A1, mu1, sigma1, A2, mu2, sigma2, d):
    return (A1 * np.exp(0.5* -(x - mu1) ** 2 / (sigma1 ** 2)))/np.sqrt(2*np.pi*sigma1**2) + (A2 * np.exp(0.5*-(x - mu2) ** 2 / (sigma2 ** 2)))/np.sqrt(2*np.pi*sigma2**2) + d

def gauss_sum2(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, d):
    return (A1 * np.exp(0.5* -(x - mu1) ** 2 / (sigma1 ** 2)))/np.sqrt(2*np.pi*sigma1**2) + (A2 * np.exp(0.5*-(x - mu2) ** 2 / (sigma2 ** 2)))/np.sqrt(2*np.pi*sigma2**2) + (A3 * np.exp(0.5* -(x - mu3) ** 2 / (sigma3 ** 2)))/np.sqrt(2*np.pi*sigma3**2) + d


frames = []
with open('dados_andre/5C_varre2.lab', 'r') as file:
    lines = file.readlines()
    data = []
    for line in lines:
        values = line.strip().split()
        values = [float(v) for v in values]
        data.append(values)
        if line == '\n':
            data = data[:-1]
            df = pd.DataFrame(data, columns=['iteracao', 'time', 'T.Piezo', 'T.Foto'])
            frames.append(df)
            data = []

df = frames[0]

T_Piezo = np.array(df['T.Piezo'])
T_Foto = np.array(df['T.Foto'])

valores_selecionados_1 = T_Piezo[(T_Piezo >= 3) & (T_Piezo <= 3.75)]
valores_selecionados_2 = T_Piezo[(T_Piezo >= 4) & (T_Piezo <= 4.30)]
valores_selecionados_3 = T_Piezo[(T_Piezo >= 6.5) & (T_Piezo <= 7.1)]
valores_selecionados_4 = T_Piezo[(T_Piezo >= 7.15) & (T_Piezo <= 7.50)]

# Calcular a média e o desvio padrão dos dados selecionados
mean_1 = np.mean(valores_selecionados_1)
mean_2 = np.mean(valores_selecionados_2)
std_dev_1 = np.std(valores_selecionados_1) 
std_dev_2 = np.std(valores_selecionados_2)
mean_3 = np.mean(valores_selecionados_3)
std_dev_3 = np.std(valores_selecionados_3) 
mean_4 = np.mean(valores_selecionados_4)
std_dev_4 = np.std(valores_selecionados_4)

# Ajustar a gaussiana apenas para o intervalo selecionado
# p0_1 = [T_Foto.max(), mean_1, std_dev_1,T_Foto[(T_Piezo >= 4) & (T_Piezo <= 4.30)].max(), mean_2, std_dev_2,5]  # Estimativas iniciais para A, mu1 e sigma1
# popt_1, covariance1 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_1)
# x_fit_1 = np.linspace(3, 4.5)  # Intervalo para o ajuste
# A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt_1[0] , popt_1[1], popt_1[2] , popt_1[3], popt_1[4], popt_1[5], popt_1[6]
# A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

# p0_2 = [T_Foto.max(), mean_3, std_dev_3,T_Foto[(T_Piezo >= 7.15) & (T_Piezo <= 7.50)].max(), mean_4, 0.5, 5]  # Estimativas iniciais para A, mu1 e sigma1
# popt_2, covariance2 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_2)
# x_fit_2 = np.linspace(6.25, 7.8)  # Intervalo para o ajuste
# A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt_2[0] , popt_2[1], popt_2[2] , popt_2[3], popt_2[4], popt_2[5], popt_2[6]
# A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo, T_Foto)

# plt.plot(x_fit_1, gauss_sum(x_fit_1, *popt_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error))
# plt.plot(x_fit_2, gauss_sum(x_fit_2, *popt_2), label=r'$A3 = %.3f \pm %.3f; \mu 3 = %.3f \pm %.3f; \sigma 3 = %.3f \pm %.3f; A4 = %.3f \pm %.3f; \mu 4 = %.3f \pm %.3f; \sigma 4 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('grafico 1')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('graf1.png')
plt.show() 

### 2 ###
'''
df = frames[1]

T_Piezo = np.array(df['T.Piezo'])
T_Foto = np.array(df['T.Foto'])

valores_selecionados_1 = T_Piezo[(T_Piezo >= 3) & (T_Piezo <= 4)]
valores_selecionados_2 = T_Piezo[(T_Piezo >= 4) & (T_Piezo <= 4.5)]
valores_selecionados_3 = T_Piezo[(T_Piezo >= 6.5) & (T_Piezo <= 7)]
valores_selecionados_4 = T_Piezo[(T_Piezo >= 7) & (T_Piezo <= 7.50)]

# Calcular a média e o desvio padrão dos dados selecionados
mean_1 = np.mean(valores_selecionados_1)
mean_2 = np.mean(valores_selecionados_2)
std_dev_1 = np.std(valores_selecionados_1) 
std_dev_2 = np.std(valores_selecionados_2)
mean_3 = np.mean(valores_selecionados_3)
std_dev_3 = np.std(valores_selecionados_3) 
mean_4 = np.mean(valores_selecionados_4)
std_dev_4 = np.std(valores_selecionados_4)

# Ajustar a gaussiana apenas para o intervalo selecionado
p0_1 = [T_Foto.max(), mean_1, std_dev_1,T_Foto[(T_Piezo >= 4) & (T_Piezo <= 4.5)].max(), mean_2, std_dev_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt_1, covariance1 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_1)
x_fit_1 = np.linspace(3, 4.5)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt_1[0] , popt_1[1], popt_1[2] , popt_1[3], popt_1[4], popt_1[5], popt_1[6]
A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

p0_2 = [T_Foto.max(), mean_3, std_dev_3,T_Foto[(T_Piezo >= 6.5) & (T_Piezo <= 7.50)].max(), mean_4, 0.5, 5]  # Estimativas iniciais para A, mu1 e sigma1
popt_2, covariance2 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_2)
x_fit_2 = np.linspace(6.25, 7.8)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt_2[0] , popt_2[1], popt_2[2] , popt_2[3], popt_2[4], popt_2[5], popt_2[6]
A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo, T_Foto)

plt.plot(x_fit_1, gauss_sum(x_fit_1, *popt_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error))
plt.plot(x_fit_2, gauss_sum(x_fit_2, *popt_2), label=r'$A3 = %.3f \pm %.3f; \mu 3 = %.3f \pm %.3f; \sigma 3 = %.3f \pm %.3f; A4 = %.3f \pm %.3f; \mu 4 = %.3f \pm %.3f; \sigma 4 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('grafico 2')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('graf1.png')
plt.show() 

### 3 ###

df = frames[2]

T_Piezo = np.array(df['T.Piezo'])
T_Foto = np.array(df['T.Foto'])

valores_selecionados_1 = T_Piezo[(T_Piezo >= 3.2) & (T_Piezo <= 3.7)]
valores_selecionados_2 = T_Piezo[(T_Piezo >= 3.7) & (T_Piezo <= 4.4)]
valores_selecionados_3 = T_Piezo[(T_Piezo >= 6.5) & (T_Piezo <= 7)]
valores_selecionados_4 = T_Piezo[(T_Piezo >= 7) & (T_Piezo <= 7.4)]

# Calcular a média e o desvio padrão dos dados selecionados
mean_1 = np.mean(valores_selecionados_1)
mean_2 = np.mean(valores_selecionados_2)
std_dev_1 = np.std(valores_selecionados_1) 
std_dev_2 = np.std(valores_selecionados_2)
mean_3 = np.mean(valores_selecionados_3)
std_dev_3 = np.std(valores_selecionados_3) 
mean_4 = np.mean(valores_selecionados_4)
std_dev_4 = np.std(valores_selecionados_4)

# Ajustar a gaussiana apenas para o intervalo selecionado
p0_1 = [T_Foto.max(), mean_1, std_dev_1,T_Foto[(T_Piezo >= 3.7) & (T_Piezo <= 4.4)].max(), mean_2, std_dev_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt_1, covariance1 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_1)
x_fit_1 = np.linspace(3, 4.5)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt_1[0] , popt_1[1], popt_1[2] , popt_1[3], popt_1[4], popt_1[5], popt_1[6]
A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

p0_2 = [T_Foto.max(), mean_3, std_dev_3,T_Foto[(T_Piezo >= 7) & (T_Piezo <= 7.4)].max(), mean_4, 0.5, 5]  # Estimativas iniciais para A, mu1 e sigma1
popt_2, covariance2 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_2)
x_fit_2 = np.linspace(6.3, 7.5)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt_2[0] , popt_2[1], popt_2[2] , popt_2[3], popt_2[4], popt_2[5], popt_2[6]
A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo, T_Foto)

plt.plot(x_fit_1, gauss_sum(x_fit_1, *popt_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error))
plt.plot(x_fit_2, gauss_sum(x_fit_2, *popt_2), label=r'$A3 = %.3f \pm %.3f; \mu 3 = %.3f \pm %.3f; \sigma 3 = %.3f \pm %.3f; A4 = %.3f \pm %.3f; \mu 4 = %.3f \pm %.3f; \sigma 4 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('grafico 3')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('graf1.png')
plt.show() 

### 4 ###

df = frames[3]

T_Piezo = np.array(df['T.Piezo'])
T_Foto = np.array(df['T.Foto'])

valores_selecionados_1 = T_Piezo[(T_Piezo >= 0) & (T_Piezo <= 0.5)]
valores_selecionados_2 = T_Piezo[(T_Piezo >= 0.7) & (T_Piezo <= 1.15)]
valores_selecionados_3 = T_Piezo[(T_Piezo >= 3.5) & (T_Piezo <= 4)]
valores_selecionados_4 = T_Piezo[(T_Piezo >= 4.2) & (T_Piezo <= 4.5)]
valores_selecionados_5 = T_Piezo[(T_Piezo >= 6.5) & (T_Piezo <= 7)]
valores_selecionados_6 = T_Piezo[(T_Piezo >= 7.3) & (T_Piezo <= 7.70)]

# Calcular a média e o desvio padrão dos dados selecionados
mean_1 = np.mean(valores_selecionados_1)
mean_2 = np.mean(valores_selecionados_2)
std_dev_1 = np.std(valores_selecionados_1) 
std_dev_2 = np.std(valores_selecionados_2)
mean_3 = np.mean(valores_selecionados_3)
std_dev_3 = np.std(valores_selecionados_3) 
mean_4 = np.mean(valores_selecionados_4)
std_dev_4 = np.std(valores_selecionados_4)
mean_5 = np.mean(valores_selecionados_5)
std_dev_5 = np.std(valores_selecionados_5)
mean_6 = np.mean(valores_selecionados_6)
std_dev_6 = np.std(valores_selecionados_6)

# Ajustar a gaussiana apenas para o intervalo selecionado
p0_1 = [T_Foto.max(), mean_1, std_dev_1,T_Foto[(T_Piezo >= 0.7) & (T_Piezo <= 1.15)].max(), mean_2, std_dev_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt_1, covariance1 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_1)
x_fit_1 = np.linspace(0, 1.2)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt_1[0] , popt_1[1], popt_1[2] , popt_1[3], popt_1[4], popt_1[5], popt_1[6]
A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

p0_2 = [T_Foto.max(), mean_3, std_dev_3,T_Foto[(T_Piezo >= 4.2) & (T_Piezo <= 4.5)].max(), mean_4, 0.5, 5]  # Estimativas iniciais para A, mu1 e sigma1
popt_2, covariance2 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_2)
x_fit_2 = np.linspace(3.4, 4.7)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt_2[0] , popt_2[1], popt_2[2] , popt_2[3], popt_2[4], popt_2[5], popt_2[6]
A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

p0_3 = [T_Foto.max(), mean_5, std_dev_5,T_Foto[(T_Piezo >= 7.3) & (T_Piezo <= 7.7)].max(), mean_6, 0.5, 5]  # Estimativas iniciais para A, mu1 e sigma1
popt_3, covariance3 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_3)
x_fit_3 = np.linspace(6.25, 7.8)  # Intervalo para o ajuste
A5 , mu5 , sigma5, A6, mu6, sigma6, d = popt_3[0] , popt_3[1], popt_3[3] , popt_3[3], popt_3[4], popt_3[5], popt_3[6]
A5_error, mu5_error, sigma5_error, A6_error, mu6_error, sigma6_error, d3_error = np.sqrt(np.diag(covariance3))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo, T_Foto)

plt.plot(x_fit_1, gauss_sum(x_fit_1, *popt_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error))
plt.plot(x_fit_2, gauss_sum(x_fit_2, *popt_2), label=r'$A3 = %.3f \pm %.3f; \mu 3 = %.3f \pm %.3f; \sigma 3 = %.3f \pm %.3f; A4 = %.3f \pm %.3f; \mu 4 = %.3f \pm %.3f; \sigma 4 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error))
plt.plot(x_fit_3, gauss_sum(x_fit_3, *popt_3), label=r'$A5 = %.3f \pm %.3f; \mu 5 = %.3f \pm %.3f; \sigma 5 = %.3f \pm %.3f; A6 = %.3f \pm %.3f; \mu 6 = %.3f \pm %.3f; \sigma 6 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A5, A5_error, mu5, mu5_error, sigma5, sigma5_error, A6, A6_error, mu6, mu6_error, sigma6, sigma6_error, d, d3_error))

plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('grafico 4')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('graf1.png')
plt.show() 

### 5 ###

df = frames[4]

T_Piezo = np.array(df['T.Piezo'])
T_Foto = np.array(df['T.Foto'])

valores_selecionados_1 = T_Piezo[(T_Piezo >= 0) & (T_Piezo <= 0.5)]
valores_selecionados_2 = T_Piezo[(T_Piezo >= 0.7) & (T_Piezo <= 1.2)]
valores_selecionados_3 = T_Piezo[(T_Piezo >= 3.5) & (T_Piezo <= 4)]
valores_selecionados_4 = T_Piezo[(T_Piezo >= 4.2) & (T_Piezo <= 4.6)]
valores_selecionados_5 = T_Piezo[(T_Piezo >= 6.5) & (T_Piezo <= 7)]
valores_selecionados_6 = T_Piezo[(T_Piezo >= 7.3) & (T_Piezo <= 7.70)]

# Calcular a média e o desvio padrão dos dados selecionados
mean_1 = np.mean(valores_selecionados_1)
mean_2 = np.mean(valores_selecionados_2)
std_dev_1 = np.std(valores_selecionados_1) 
std_dev_2 = np.std(valores_selecionados_2)
mean_3 = np.mean(valores_selecionados_3)
std_dev_3 = np.std(valores_selecionados_3) 
mean_4 = np.mean(valores_selecionados_4)
std_dev_4 = np.std(valores_selecionados_4)
mean_5 = np.mean(valores_selecionados_5)
std_dev_5 = np.std(valores_selecionados_5)
mean_6 = np.mean(valores_selecionados_6)
std_dev_6 = np.std(valores_selecionados_6)

# Ajustar a gaussiana apenas para o intervalo selecionado
p0_1 = [T_Foto.max(), mean_1, std_dev_1,T_Foto[(T_Piezo >= 4.2) & (T_Piezo <= 4.6)].max(), mean_2, std_dev_2,5]  # Estimativas iniciais para A, mu1 e sigma1
popt_1, covariance1 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_1)
x_fit_1 = np.linspace(0, 1.3)  # Intervalo para o ajuste
A1 , mu1 , sigma1, A2, mu2, sigma2, d = popt_1[0] , popt_1[1], popt_1[2] , popt_1[3], popt_1[4], popt_1[5], popt_1[6]
A1_error, mu1_error, sigma1_error, A2_error, mu2_error, sigma2_error, d1_error = np.sqrt(np.diag(covariance1))

p0_2 = [T_Foto.max(), mean_3, std_dev_3,T_Foto[(T_Piezo >= 7.3) & (T_Piezo <= 7.7)].max(), mean_4, 0.5, 5]  # Estimativas iniciais para A, mu1 e sigma1
popt_2, covariance2 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_2)
x_fit_2 = np.linspace(3.4, 4.7)  # Intervalo para o ajuste
A3 , mu3 , sigma3, A4, mu4, sigma4, d = popt_2[0] , popt_2[1], popt_2[2] , popt_2[3], popt_2[4], popt_2[5], popt_2[6]
A3_error, mu3_error, sigma3_error, A4_error, mu4_error, sigma4_error, d2_error = np.sqrt(np.diag(covariance2))

p0_3 = [T_Foto.max(), mean_5, std_dev_5,T_Foto[(T_Piezo >= 7.3) & (T_Piezo <= 7.7)].max(), mean_6, 0.5, 5]  # Estimativas iniciais para A, mu1 e sigma1
popt_3, covariance3 = curve_fit(gauss_sum, T_Piezo, T_Foto, p0=p0_3)
x_fit_3 = np.linspace(6.25, 7.8)  # Intervalo para o ajuste
A5 , mu5 , sigma5, A6, mu6, sigma6, d = popt_3[0] , popt_3[1], popt_3[3] , popt_3[3], popt_3[4], popt_3[5], popt_3[6]
A5_error, mu5_error, sigma5_error, A6_error, mu6_error, sigma6_error, d3_error = np.sqrt(np.diag(covariance3))

# Plot dos dados e da gaussiana ajustada
plt.figure(figsize=(12, 6))

plt.plot(T_Piezo, T_Foto)

plt.plot(x_fit_1, gauss_sum(x_fit_1, *popt_1), label=r'$A1 = %.3f \pm %.3f; \mu 1 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; A2 = %.3f \pm %.3f; \mu 2 = %.3f \pm %.3f; \sigma 2 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A1, A1_error, mu1, mu1_error, sigma1, sigma1_error, A2, A2_error, mu2, mu2_error, sigma2, sigma2_error, d, d1_error))
plt.plot(x_fit_2, gauss_sum(x_fit_2, *popt_2), label=r'$A3 = %.3f \pm %.3f; \mu 3 = %.3f \pm %.3f; \sigma 3 = %.3f \pm %.3f; A4 = %.3f \pm %.3f; \mu 4 = %.3f \pm %.3f; \sigma 4 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A3, A3_error, mu3, mu3_error, sigma3, sigma3_error, A4, A4_error, mu4, mu4_error, sigma4, sigma4_error, d, d2_error))
plt.plot(x_fit_3, gauss_sum(x_fit_3, *popt_3), label=r'$A5 = %.3f \pm %.3f; \mu 5 = %.3f \pm %.3f; \sigma 5 = %.3f \pm %.3f; A6 = %.3f \pm %.3f; \mu 6 = %.3f \pm %.3f; \sigma 6 = %.3f \pm %.3f; d = %.3f \pm %.3f$' %(A5, A5_error, mu5, mu5_error, sigma5, sigma5_error, A6, A6_error, mu6, mu6_error, sigma6, sigma6_error, d, d3_error))


plt.xlabel('T.Piezo')
plt.ylabel('T.Foto')
plt.title('grafico 5')

x_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_values_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.xticks(x_values_to_display)
plt.yticks(y_values_to_display)

plt.legend(fontsize='small')
plt.savefig('graf1.png')
plt.show() 
'''
