#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importando bibliotecas
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


#Para importar datos de tracker


# In[107]:


#Definición de variables:
m1, m2, m3, m4 = 0.6, 0.6, 0.3, 0.3
L1, L2, L3, L4 = 1  , 2,  1,  2
#Hay 4 posibles casos. Inicialmente se utilizará un ángulo pequeño


# In[116]:


#Encontrando los ángulos donde sen(x) = x con un margen de error del 3%:
x = np.linspace(0,1,100)
error = []
margen = 0.03 #Pd: este lo podemos cambiar si es necesario trabajar ángulos más grandes (Mencionar los cambios en el informe)
for i in range(0, len(x)):
    angulo = math.sin(x[i])
    s = (x[i]*margen) #Calculando el valor mínimo de sen(x) para que el margen de error sea de 3%
    error.append(x[i]-angulo)
    if error[i] > s:
        print("Es posible aproximar hasta que el ángulo valga", x[i-1])
        break


# In[140]:


#Caso 1, con m1 y L1:
time = np.linspace(0, 1, 1000)
a = np.zeros(len(time))  #aceleración angular
w = np.zeros(len(time))  #velocidad angular
T = np.zeros(len(time))  #Tensión
angulo = np.zeros(len(time))
angulo[0] = 0.1
g = 9.78
for i in range(0,len(time)-1):
    cambio = time[i+1]-time[i]
    a[i]   = g*angulo[i] / L1
    T[i]   = -m1*w[i]*w[i]*L1 + m1*g*math.cos(angulo[i])
    w[i+1] = w[i] + a[i] * (cambio)
    angulo[i+1] = angulo[i] + w[i]*(cambio) + a[i]* ((cambio**2) / 2)
    if T[i]<0:
        print("El tiempo es de", time[i])
        print("La aceleración angular máxima es de", a[i])
        print("La velocidad angular maxima es de", w[i])
        print("El ángulo máximo es de", angulo[i])
        break
    
    


# In[ ]:




