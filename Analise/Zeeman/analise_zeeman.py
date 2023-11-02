import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

############# Functions #############
def linear(x, a, b):
    return a*x + b

def plotter(I, dif_pos, x_Err, y_Err, name):
    plt.figure(figsize=(8, 6))
    plt.errorbar(I, dif_pos, xerr = x_Err, yerr = y_Err, fmt='.', color='black')
    plt.xlabel('I [A]', fontsize=14)
    plt.ylabel('pos [x10^-3 m]', fontsize=14)
    plt.grid()
    #plt.savefig(name + '.png')
    plt.show()

def plotter_fit(I, dif_pos, x_Err, y_Err, name, a, b, erro_a, erro_b):
    plt.figure(figsize=(8, 6))
    plt.errorbar(I, dif_pos, xerr = x_Err, yerr = y_Err, fmt='.', color='black')
    plt.plot(I, linear(np.array(I), a, b), color='red')
    plt.xlabel('I [A]', fontsize=14)
    plt.ylabel('pos [x10^-3 m]', fontsize=14)
    plt.title(name, fontsize=16)
    plt.text(0.1, 0.90, f'a : {round(a, 5)} +- {round(erro_a, 5)}', transform=plt.gca().transAxes, color='black')
    plt.text(0.1, 0.85, f'b : {round(b, 5)} +- {round(erro_b, 5)}', transform=plt.gca().transAxes, color='black')
    plt.grid()
    #plt.savefig(name + '.png')
    plt.show()

################## Constants ##################
erro_I = 0.1
erro_pos = 0.005

################## Data ##################
### Montagem Longitudinal 
# sigma- 
I_L_sigma_menos = [0.4, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 19.5] # A
pos_L_sigma_menos = [6.97, 6.96, 6.96, 6.99, 7, 7.03, 7.06, 7.08, 7.1] # x10^-3 m

# sigma+ 
I_L_sigma_mais = [0.4, 2.5, 5, 7.5, 9.9, 12.5, 15, 17.5, 19.6] # A
pos_L_sigma_mais = [6.94, 6.94, 6.94, 6.93, 6.91, 6.9, 6.89, 6.87,6.87] # x10^3 m

### Montagem Transversal 
# primeira linha 
I_T_1a = [0.3, 2.5,5,7.5,10,12.5,15,17.5,19.7] # A
pos_T_1a = [6.95, 6.95, 6.92, 6.90, 6.88, 6.87, 6.86, 6.86, 6.85] # x10^-3 m

# segunda linha 
I_T_2a = [0.3, 2.5,5,7.5,10,12.5,15,17.5,19.8] # A
pos_T_2a = [6.93, 6.93, 6.92, 6.90, 6.90, 6.90, 6.88, 6.88, 6.88] # x10^-3 m

# terceira linha 
I_T_3a = [0.3, 2.5,5,7.5,10,12.5,15,17.5,19.8] # A
pos_T_3a = [6.94, 6.94, 6.96, 6.97, 6.98, 7.00, 7.00, 7.00, 7.02] # x10^-3 m

# quarta linha 
I_T_4a = [0.3, 2.5,5,7.5,10,12.5,15,17.5,19.8] # A
pos_T_4a = [6.93, 6.93, 6.95, 6.95, 6.97, 6.98, 7.02, 7.09, 7.12] # x10^-3 m

# pi 
I_T_pi = [0.4, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 19.6] # A
pos_pi = [6.94, 6.94, 6.94, 6.93, 6.91, 6.9, 6.89, 6.87, 6.87] # x10^-3 m

# sigma_menos 
I_T_sigma_menos = [0.5, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 19.4] # A
pos_T_sigma_menos = [6.880, 6.880, 6.920, 6.940, 6.940, 6.950, 6.950, 6.980, 7.010] # x10^-3 m~

# sigma_mais 
I_T_sigma_mais = [0.5, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 19.4] # A
pos_T_sigma_mais = [6.91, 6.91, 6.90, 6.89, 6.88, 6.87, 6.86, 6.86, 6.85] # x10^-3 m


I = [I_L_sigma_menos, I_L_sigma_mais, I_T_1a, I_T_2a, I_T_3a, I_T_4a, I_T_pi, I_T_sigma_menos, I_T_sigma_mais]
Pos = [pos_L_sigma_menos, pos_L_sigma_mais, pos_T_1a, pos_T_2a, pos_T_3a, pos_T_4a, pos_pi, pos_T_sigma_menos, pos_T_sigma_mais]
names = ['Longitudinal sigma-', 'Longitudinal sigma+', 'Transversal 1a', 'Trasnversal 2a',
         'Transversal 3a', 'Transversal 4a', 'Transverl pi', 'Transversal sigma-', 'Transversal sigma+']
################## Analysis ###########################

##### fit dos dados
for i in range(len(I)):
    pos_new = []
    for p in Pos[i]:
        pos_new.append(abs(p - Pos[i][0]))

    # plot
    plotter(I[i], pos_new, erro_I, erro_pos * 2, names[i])

    # fit
    params, cov = curve_fit(linear, I[i], pos_new)
    a, b = params
    erro_a, erro_b = np.sqrt(np.diag(cov))

    plotter_fit(I[i], pos_new, erro_I, erro_pos * 2, names[i], a, b, erro_a, erro_b)

##### Obter uma expressão para o campo magnético em função da corrente 
""" B = 0.04 * I"""
I_values = [5, 10, 15, 20] # A
B_values = [0.2, 0.4, 0.6, 0.8] # T

# fit
"""params, cov = curve_fit(linear, I_values, B_values)
a, b = params
erro_a, erro_b = np.sqrt(np.diag(cov))
plotter_fit(I_values, B_values, 0, 0, 'Campo magnético em função da corrente', a, b, erro_a, erro_b)"""
