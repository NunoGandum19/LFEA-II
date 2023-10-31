import numpy as np
import matplotlib.pyplot as plt

#FCC
miller_options = [[1,0,0], [1,1,0], [1, 1, 1], [2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1]]

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
for list in miller_options:
    print('indices de Miller: ',list, 'Distância: ', d_cubic(list[0], list[1], list[2], a_al))
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
