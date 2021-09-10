#!/usr/bin/env python
# coding: utf-8

# # Código - Estimación de la gravedad a partir del movimiento de un péndulo
# Brayan Amorocho - Santiago Montes - Juliana Andrade
# 

# In[106]:


#Importando bibliotecas
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


# In[90]:


#Para importar datos de tracker


# In[91]:


#Definición de variables:
m1, m2, m3, m4 = 0.6, 0.6, 0.3, 0.3
L1, L2, L3, L4 = 1  , 2,  1,  2
#m = [0.6, 0.6, 0.3, 0.3]
#Hay 4 posibles casos. Inicialmente se utilizará un ángulo pequeño


# In[94]:


#Fuentes bonitas para las gráficas futuras
font = {"family" : "serif",
        "color"  : "darkred",
        "weight" : "bold",
        "size"   : 16,
        }

font2 = {"family" : "serif",
         "color"  : "black",
         "weight" : "normal",
         "size"   : 12,
         }


# In[92]:


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


# In[93]:


#Caso 1, con m1 y L1:
time = np.linspace(0, 10, 100000)
a = np.zeros(len(time))  #aceleración angular
w = np.zeros(len(time))  #velocidad angular
T = np.zeros(len(time))  #Tensión
angulo = np.zeros(len(time))
angulo[0] = 0.42
g = 9.78
for i in range(0,len(time)-1):
    cambio = time[i+1]-time[i]
    a[i]   = g*angulo[i] / L1
    T[i]   = -m1*w[i]*w[i]*L1 + m1*g*math.cos(angulo[i])
    w[i+1] = w[i] + a[i] * (cambio)
    angulo[i+1] = angulo[i] - w[i]*(cambio) + a[i]* ((cambio**2) / 2)
angulo_grad = angulo * 180 / math.pi #Pasar a grados para hacer más intuitiva la gráfica
    
    
    


# In[110]:


"""
Conjunto de gráficas para el caso 1, en un mundo sin fricción
"""
#Angulo en grados
plt.figure()
plt.plot(time,angulo_grad, "k")
plt.title("Movimiento gradual", fontdict = font)
plt.ylabel(r"Ángulo $\theta°$", fontdict = font2)
plt.xlabel("Tiempo ($s$)", fontdict=font2)
plt.grid(linestyle="--")

#Angulo en radianes
plt.figure()
plt.plot(time,angulo, "k")
plt.title("Movimiento gradual", fontdict = font)
plt.ylabel(r"Ángulo $\theta~(Rad)$", fontdict = font2)
plt.xlabel("Tiempo ($s$)", fontdict=font2)
plt.grid(linestyle="--")

#Movimiento Angular
plt.figure()
plt.plot(time,w, "g")
plt.title("Velocidad Angular en función del tiempo", fontdict = font)
plt.ylabel(r"Velocidad angular $\theta~(Rad)$", fontdict = font2)
plt.xlabel("Tiempo ($s$)", fontdict=font2)
plt.grid(linestyle="--")

#Comparación
plt.figure()
plt.plot(time,angulo, "k", label="Ángulo")
plt.plot(time,w, "g", label = "Velocidad Angular")
plt.legend(loc = "best")
plt.title("Movimiento gradual", fontdict = font)
plt.xlabel("Tiempo ($s$)", fontdict=font2)
plt.grid(linestyle="--")


# In[116]:


#Caso 2 con m2 y L2:
time = np.linspace(0, 10, 100000)
a = np.zeros(len(time))  #aceleración angular
w = np.zeros(len(time))  #velocidad angular
T = np.zeros(len(time))  #Tensión
angulo = np.zeros(len(time))
angulo[0] = 0.42
g = 9.78
for i in range(0,len(time)-1):
    cambio = time[i+1]-time[i]
    a[i]   = g*angulo[i] / L2
    T[i]   = -m2*w[i]*w[i]*L2 + m2*g*math.cos(angulo[i])
    w[i+1] = w[i] + a[i] * (cambio)
    angulo[i+1] = angulo[i] - w[i]*(cambio) + a[i]* ((cambio**2) / 2)
angulo_grad = angulo * 180 / math.pi #Pasar a grados para hacer más intuitiva la gráfica


# In[117]:


"""
Conjunto de gráficas para el caso 2, en un mundo sin fricción
"""
#Angulo en grados
plt.figure()
plt.plot(time,angulo_grad, "k")
plt.title("Movimiento gradual", fontdict = font)
plt.ylabel(r"Ángulo $\theta°$", fontdict = font2)
plt.xlabel("Tiempo ($s$)", fontdict=font2)
plt.grid(linestyle="--")

#Angulo en radianes
plt.figure()
plt.plot(time,angulo, "k")
plt.title("Movimiento gradual", fontdict = font)
plt.ylabel(r"Ángulo $\theta~(Rad)$", fontdict = font2)
plt.xlabel("Tiempo ($s$)", fontdict=font2)
plt.grid(linestyle="--")

#Movimiento Angular
plt.figure()
plt.plot(time,w, "g")
plt.title("Velocidad Angular en función del tiempo", fontdict = font)
plt.ylabel(r"Velocidad angular $\theta~(Rad)$", fontdict = font2)
plt.xlabel("Tiempo ($s$)", fontdict=font2)
plt.grid(linestyle="--")

#Comparación
plt.figure()
plt.plot(time,angulo, "k", label="Ángulo")
plt.plot(time,w, "g", label = "Velocidad Angular")
plt.legend(loc = "best")
plt.title("Movimiento gradual", fontdict = font)
plt.xlabel("Tiempo ($s$)", fontdict=font2)
plt.grid(linestyle="--")


# In[99]:





# In[104]:





# In[103]:




