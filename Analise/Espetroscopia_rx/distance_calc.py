import numpy as np
import matplotlib.pyplot as plt

#FCC
miller_options = [[1,0,0], [1,1,0], [1, 1, 1], [2, 0, 0], [2, 1, 0], [2, 1, 1], 
                  [2, 2, 0], [2, 2, 1], [2,2,2], [3,0,0], [3,1,0],[3,1,1],
                  [3,2,0],[3,2,2],[3,3,0],[3,3,1],[3,3,2],[3,3,3],[4,0,0],
                  [4,1,1],[4,1,0],[4,2,0],[4,2,1],[4,2,2],[4,3,0],[4,3,1],
                  [5,0,0],[5,1,0],[5,1,1],[5,2,0],[5,2,1],[5,2,2],[5,3,0],
                    [5,3,1],[5,3,2],[5,3,3],[6,0,0],[6,1,0],[6,1,1],[6,2,0],
                    [6,2,1],[6,2,2],[6,3,0],[6,3,1],[6,3,2],[6,3,3],[7,0,0],
                    [7,1,0],[7,1,1],[7,2,0],[7,2,1],[7,2,2],[7,3,0],[7,3,1],
                    [7,3,2],[7,3,3],[8,0,0],[8,1,0],[8,1,1],[8,2,0],[8,2,1],
                    [8,2,2],[8,3,0],[8,3,1],[8,3,2],[8,3,3],[9,0,0],[9,1,0],
                    [9,1,1],[9,2,0],[9,2,1],[9,2,2],[9,3,0],[9,3,1],[9,3,2],
                    [9,3,3],[10,0,0],[10,1,0],[10,1,1],[10,2,0],[10,2,1],
                    [10,2,2],[10,3,0],[10,3,1],[10,3,2],[10,3,3],[11,0,0],
                    [11,1,0],[11,1,1],[11,2,0],[11,2,1],[11,2,2],[11,3,0]]

#a (lattice parameter = aresta in armstrong)
a_nacl = 563
a_al = 405
a_si = 543
a_lif = 402.6 #https://princetonscientific.com/materials/substrates-wafers/lithium-fluoride/
a_saphire = 475.8 #https://princetonscientific.com/materials/substrates-wafers/aluminium-oxide-sapphire/
c_saphire = 129.91 #https://princetonscientific.com/materials/substrates-wafers/aluminium-oxide-sapphire/
a_hopg = b = 246 #https://www.hqgraphene.com/HOPG.php
c_hopg = 667 
# alpha = 90
# beta = 90
# gamma = 120

#d (interplanar spacing)
#https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119961468.app3

def d_cubic(h,k,l,a): return a/np.sqrt(h**2+k**2+l**2)

def d_hexagonal(h,k,l,a,c): return 1/(np.sqrt(4/3*(h**2+h*k+k**2)/a**2+l**2/c**2))

#NaCl
print('NaCl')
for list in miller_options:
    print('indices de Miller: ',list, 'Distância: ',d_cubic(list[0], list[1], list[2], a_nacl))
print('\n')

# Al
print('Al')
opcoes_al_dis = []
for list in miller_options:
    print('indices de Miller: ',list, 'Distância: ', d_cubic(list[0], list[1], list[2], a_al))
    opcoes_al_dis = opcoes_al_dis + [d_cubic(list[0], list[1], list[2], a_al)]
print('\n')

print('Si')
# Si
for list in miller_options:
    print('indices de Miller: ',list, 'Distância: ', d_cubic(list[0], list[1], list[2], a_si))
print('\n')

print('LiF')
# LiF
for list in miller_options:
    print('indices de Miller: ',list, 'Distância: ', d_cubic(list[0], list[1], list[2], a_lif))
print('\n')

print('Saphire')
# Saphire
for list in miller_options:
    print('indices de Miller: ',list, 'Distância: ', d_hexagonal(list[0], list[1], list[2], a_saphire, c_saphire))
print('\n')

print('HOPG')
# HOPG
for list in miller_options:
    print('indices de Miller: ',list, 'Distância: ', d_hexagonal(list[0], list[1], list[2], a_hopg, c_hopg))


import numpy as np

arr = np.full(len(opcoes_al_dis), 212)
error = np.abs(np.array(opcoes_al_dis) - arr)
# get the index of the smallest value in the error array
min_index = np.argmin(error)

# print the index
print('min error is with d = ',opcoes_al_dis[min_index], 'with miller index: ', miller_options[min_index])

print('percentual error is ', error[min_index]/opcoes_al_dis[min_index]*100, '%')

    