#!/usr/bin/env python
# coding: utf-8

# # Código - Estimación de la gravedad a partir del movimiento de un péndulo
# Brayan Amorocho - Santiago Montes - Juliana Andrade
# 



#Importando bibliotecas
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd




#Para importar datos de tracker



#Definición de variables:
m1, m2, m3, m4 = 0.6, 0.6, 0.3, 0.3
L1, L2, L3, L4 = 1  , 2,  1,  2
m = [0.6, 0.6, 0.3, 0.3]
L = [ 1, 2, 1, 2]
#Hay 4 posibles casos. Inicialmente se utilizará un ángulo pequeño




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



time = np.linspace(0, 10, 100000)
a, T, w = [], [], [] #Conjuntos vacíos para la aceleración y el movimiento angular y la tensión, se usarán más adelante.
angulo, angulo_grad = [], []
g = 9.78
for j in range(0,4): #Para hacer de una vez los 4 casos
    
    #Creando listas vacías para cada variable
    a.append(np.zeros(len(time)))
    w.append(np.zeros(len(time)))
    T.append(np.zeros(len(time)))
    angulo.append(np.zeros(len(time)))
    angulo[j][0] = 0.42
    #Aplicando las fórmulas
    for i in range(0,len(time)-1):
        cambio = time[i+1]-time[i]
        a[j][i]   = g*angulo[j][i] / L[j]
        T[j][i]   = -m[j]*w[j][i]*w[j][i]*L[j] + m[j]*g*math.cos(angulo[j][i])
        w[j][i+1] = w[j][i] + a[j][i] * (cambio)
        angulo[j][i+1] = angulo[j][i] - w[j][i]*(cambio) + a[j][i]* ((cambio**2) / 2)
    angulo_grad.append(angulo[j] * 180 / math.pi) #Pasar a grados para hacer más intuitiva la gráfica

    


def conjunto_graficas(n):
    """
    Conjunto de gráficas, en un mundo sin fricción
    """
    #Angulo en grados
    plt.figure()
    plt.plot(time,angulo_grad[n], "k")
    plt.title("Movimiento gradual", fontdict = font)
    plt.ylabel(r"Ángulo $\theta°$", fontdict = font2)
    plt.xlabel("Tiempo ($s$)", fontdict=font2)
    plt.grid(linestyle="--")

    #Angulo en radianes
    plt.figure()
    plt.plot(time,angulo[n], "k")
    plt.title("Movimiento gradual", fontdict = font)
    plt.ylabel(r"Ángulo $\theta~(Rad)$", fontdict = font2)
    plt.xlabel("Tiempo ($s$)", fontdict=font2)
    plt.grid(linestyle="--")

    #Movimiento Angular
    plt.figure()
    plt.plot(time,w[n], "g")
    plt.title("Velocidad Angular en función del tiempo", fontdict = font)
    plt.ylabel(r"Velocidad angular $\theta~(Rad)$", fontdict = font2)
    plt.xlabel("Tiempo ($s$)", fontdict=font2)
    plt.grid(linestyle="--")

    #Comparación
    plt.figure()
    plt.plot(time,angulo[n], "k", label="Ángulo")
    plt.plot(time,w[n], "g", label = "Velocidad Angular")
    plt.legend(loc = "best")
    plt.title("Movimiento gradual", fontdict = font)
    plt.xlabel("Tiempo ($s$)", fontdict=font2)
    plt.grid(linestyle="--")
    
    
    


conjunto_graficas(0)

conjunto_graficas(1)

conjunto_graficas(2)

conjunto_graficas(3)



