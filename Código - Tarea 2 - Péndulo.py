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
#Los documentos de excel deberían estar en la misma carpeta (O copiar la ruta de acceso a ellos)
prueba = []
#0.5m
#5 grados
prueba.append(pd.read_excel("Angulos de 5, 300mL, 0.5m.xlsx", usecols=("A:BA"), skiprows = 4))
prueba.append(pd.read_excel("Ángulos de 5 grados, 400mL, 0.5M de longitud.xlsx", usecols=("A:BA"), skiprows = 4))

#12 grados
prueba.append(pd.read_excel("Ángulos de 12 grados, 300mL, 0.5m de longitud.xlsx", usecols=("A:BA"), skiprows = 4))
prueba.append(pd.read_excel("Ángulos de 12 grados, 400mL, 0.5M de longitud.xlsx", usecols=("A:BA"), skiprows = 4))

#20 grados
prueba.append(pd.read_excel(r"Ángulo de 20°, 300mL, 0.5m.xlsx", usecols=("A:BA"), skiprows = 1))
prueba.append(pd.read_excel(r"Ángulo de 20°, 400mL y 0.5 metro de longitud.xlsx", usecols=("A:BA"), skiprows = 1))

#33 grados
prueba.append(pd.read_excel(r"300ml 32°(50cm).xlsx", usecols=("A:BA"), skiprows = 1))
prueba.append(pd.read_excel(r"Ángulo de 33°, 400mL, 0,5m de longitud.xlsx", usecols=("A:BA"), skiprows = 4))

#1 metro
#5 grados
prueba.append(pd.read_excel(r"Ángulos de 5 grados, 300mL, 1m de longitud.xlsx", usecols=("A:BA"), skiprows = 4))
prueba.append(pd.read_excel(r"Ángulos de 5 Grados, 400mL, 1m de longitud.xlsx", usecols=("A:Z"), skiprows = 5))

#12 grados
prueba.append(pd.read_excel(r"Ángulos de 12 Grados 300mL, 1m de longitud.xlsx", usecols=("A:AV"), skiprows = 5))
prueba.append(pd.read_excel(r"Ángulos de 12 Grados 400mL, 1m de longitud.xlsx", usecols=("A:BA"), skiprows = 5))
#20 grados
prueba.append(pd.read_excel(r"Ángulos de 20 Grados, 300mL, 1m de longitud.xlsx", usecols=("A:AQ"), skiprows = 5))
prueba.append(pd.read_excel(r"Ángulos de 20 Grados, 400mL, 1m de longitud.xlsx", usecols=("A:AB"), skiprows = 5))

#33 grados
prueba.append(pd.read_excel(r"Ángulo de 33°, 300mL y 1 metro de longitud.xlsx", usecols=("A:BA"), skiprows = 3))
prueba.append(pd.read_excel(r"Ángulo de 33°, 400mL y 1 metro.xlsx", usecols=("A:BA"), skiprows = 4))


#Definición de variables:
an = [  5,   5,  12,  12,  20,   20,   33,  33,    5,    5,   12,   12,   20,  20,    33,   33]
m  = [0.3, 0.4, 0.3, 0.4, 0.3,  0.4,  0.3, 0.4,  0.3,  0.4,  0.3,  0.4,  0.3,  0.4,  0.3,  0.4]
L  = [0.5, 0.5, 0.5, 0.5, 0.5,  0.5,  0.5, 0.5, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05]



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
        print("Es posible aproximar hasta que el ángulo valga {} radianes.".format((round(x[i-1],2))))
        break


##Datos del modelo teórico
time = np.arange(0, 7, 0.00001)
a, T, w = [], [], [] #Conjuntos vacíos para la aceleración y el movimiento angular y la tensión, se usarán más adelante.
angulo, angulo_grad = [], []
g = 9.78
for j in range(0,len(an)): #Para hacer de una vez los 16 casos
    
    #Creando listas vacías para cada variable
    a.append(np.zeros(len(time)))
    w.append(np.zeros(len(time)))
    T.append(np.zeros(len(time)))    
    angulo.append(np.zeros(len(time)))
    angulo[j][0] = an[j] * math.pi / 180
    
    #Aplicando las fórmulas que están en el informe 
    for i in range(0,len(time)-1):
        cambio = time[i+1]-time[i]
        a[j][i]   = g*angulo[j][i] / L[j] #Aceleración angular
        #T[j][i]   = -m[j]*w[j][i]*w[j][i]*L + m[j]*g*math.cos(angulo[j][i]) #Tensión (No usada)
        w[j][i+1] = (w[j][i] + a[j][i] * (cambio)) #Velocidad angular
        angulo[j][i+1] = (angulo[j][i] - w[j][i]*(cambio) + a[j][i]* ((cambio**2) / 2)) #angulo en radianes
    angulo_grad.append(angulo[j] * 180 / math.pi) #Pasar a grados para hacer más intuitiva la gráfica

#Puede que redondear los valores del tiempo nos sea útil más tarde ;)
tiempo_aprox = [round(elem, 3) for elem in time ]


#Graficando el modelo teórico
def conjunto_graficas(n): #Teóricas
   
    #Para el título
    medida = round(m[n] *1000)
    #Angulo en grados
    plt.figure()
    plt.plot(time,angulo_grad[n], "k")
    plt.title(r"Movimiento gradual $(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font) #Acá lo que hago es cambiar el nombre del eje
    plt.ylabel(r"Ángulo $\theta° = {}$".format(an[n]), fontdict = font2)
    plt.xlabel("Tiempo ($s$)", fontdict=font2)
    plt.grid(linestyle="--")

    #Angulo en radianes
    plt.figure()
    plt.plot(time,angulo[n], "k")
    plt.title(r"Movimiento Angular $(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font)
    plt.ylabel(r"Ángulo $\theta~(Rad)$", fontdict = font2)
    plt.xlabel("Tiempo ($s$)", fontdict=font2)
    plt.grid(linestyle="--")

    #Movimiento Angular
    plt.figure()
    plt.plot(time,w[n], "g")
    plt.title(r"Velocidad Angular $(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font)
    plt.ylabel(r"Velocidad angular $\theta~(Rad)$", fontdict = font2)
    plt.xlabel("Tiempo ($s$)", fontdict=font2)
    plt.grid(linestyle="--")

        
      

#Análisis de datos de tracker

#Definición de variables experimentales 1

def obtencion_datos(n): #n es el numero de prueba
        
    tiempo1, angulo1, angulo_rad1, velocidad1 = [], [], [], [] #Acá iremos agregando arrays para cada experimento

    bucle = int(prueba[n].shape[1]/5) +1 #Una forma que encontré para hallar la cantidad de columnas en cada documento
    
    tiempo1.append(np.array(prueba[n]["t"])) #Vamos agregando cada columna como arrays
    angulo1.append(np.array(prueba[n]["θ"]))
    velocidad1.append(np.array(prueba[n]["ω"])*math.pi / 180 *-1) #Hay que pasarlo a radianes, multiplicamos por -1 pq el sistema de referencia en tracker está "invertido"
    angulo_rad1.append(np.array(prueba[n]["θ"]) / 180 * math.pi)
    for i in range(1,bucle): #Acá nos sirve lo de la cantidad de columnas
        tiempo_prueba = (np.array(prueba[n]["t.{}".format(i)])) #Una forma de seguir agregando arrays pero automática, sin importar qué tantas columnas haya
        angulo_prueba = (np.array(prueba[n]["θ.{}".format(i)]))
        velocidad_prueba = (np.array(prueba[n]["ω.{}".format(i)])*math.pi / 180 *-1) #Pasamos a radianes y multiplico por -1 pq tracker nos da el sistema de referencia invertido
        angulo_rad_prueba = (np.array(prueba[n]["θ.{}".format(i)]/ 180 *math.pi)) #Pasamos a radianes

        tiempo1.append(np.array([x for x in tiempo_prueba if pd.isnull(x) == False])) #Con esto lo que hago es eliminar las filas vacías en cada columna, por eso era importante que la de mayor filas fuese la primera
        angulo1.append(np.array([x for x in angulo_prueba if pd.isnull(x) == False]))
        velocidad1.append(np.array([x for x in velocidad_prueba if pd.isnull(x) == False]))
        angulo_rad1.append(np.array([x for x in angulo_rad_prueba if pd.isnull(x) == False]))
    
    def grafica_experimento(n): #Acá graficamos los datos solos
        color = ["b", "g", "r", "c", "m", "y", "b", "g", "r", "m"]
        medida = round(m[n]*1000)
    
        plt.figure()
        for i in range(1,len(tiempo1)-1): #Pa poner todos en una sola gráfica
            plt.plot(tiempo1[i],angulo1[i], "--{}".format(color[i]), label="Experimento {}".format(i), alpha=0.7)
        plt.title(r"Movimiento gradual: Experimento $(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font)
        plt.ylabel(r"Ángulo $\theta°$", fontdict = font2)
        plt.xlabel("Tiempo ($s$)", fontdict=font2)
        plt.legend(loc = "best", bbox_to_anchor=(0.9, 0.5, 0.5, 0.5))
        plt.grid(linestyle="--")
        plt.show()

        #Angulo en radianes
        plt.figure()
        for i in range(1,len(tiempo1)-1):
            plt.plot(tiempo1[i],angulo_rad1[i], "--{}".format(color[i]), label="Experimento {}".format(i), alpha=0.7)
        plt.title(r"Movimiento Angular: Experimento $(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font)
        plt.ylabel(r"Ángulo $\theta~(Rad)$", fontdict = font2)
        plt.xlabel("Tiempo ($s$)", fontdict=font2)
        plt.legend(loc = "best", bbox_to_anchor=(0.9, 0.5, 0.5, 0.5))
        plt.grid(linestyle="--")
        plt.show()

        #Movimiento Angular
        plt.figure()
        for i in range(1,len(tiempo1)-1):
            plt.plot(tiempo1[i],velocidad1[i], "--{}".format(color[i]), label="Experimento {}".format(i), alpha=0.7)
        plt.title(r"Velocidad Angular: Experimento$(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font)
        plt.ylabel(r"Velocidad angular $\theta~(Rad)$", fontdict = font2)
        plt.xlabel("Tiempo ($s$)", fontdict=font2)
        plt.legend(loc = "best", bbox_to_anchor=(0.9, 0.5, 0.5, 0.5))
        plt.grid(linestyle="--")
        plt.show()

    
    def grafica_comparativa(n,time): #Acá comparamos el promedio con la teoría
        #Adaptando al tamaño de la grafica de experimentos
        pos = (np.where(tiempo_aprox == tiempo1[0][-1]))    
        recorte = pos[0][0]-500
        time2= time[:recorte]
        angulo_grad2 = angulo_grad[n][:recorte]
        w2 = w[n][:recorte]
        angulo2 = angulo[n][:recorte]
        
        #Calculando el margen de error en cada uno
        
        #Creamos unas listas que serán los puntos donde se trazan las rayas
        paso = int(len(tiempo1[0])/12)
        t_err = tiempo1[0][::paso-1]
        ang_err = angulo1[0][::paso-1]
        ang_rad_err = angulo_rad1[0][::paso-1]
        vel_err = velocidad1[0][::paso-1]
        
        # Necesitamos evaluar el valor máximo y mínimo en dichos puntos
    
        new_angle = np.transpose(angulo1)[::paso-1]
        new_angle_rad = np.transpose(angulo_rad1)[::paso-1]
        new_vel = np.transpose(velocidad1)[::paso-1]
        
        error_ang_up, error_rad_up, error_vel_up = [],[],[]
        error_ang_down, error_rad_down, error_vel_down = [],[],[]
        
        
        #Hacemos la resta de valor max o min - valor promedio para hallar el error
        for i in range(0,len(new_angle)):
            error_ang_up.append(abs(round(max(new_angle[i]) - ang_err[i],3)))
            error_ang_down.append(abs(round(min(new_angle[i]) - ang_err[i],3)))
            error_rad_up.append(abs(round(max(new_angle_rad[i]) - ang_rad_err[i],3)))
            error_rad_down.append(abs(round(min(new_angle_rad[i]) - ang_rad_err[i],3)))
            error_vel_up.append(abs(round(max(new_vel[i]) - vel_err[i],3)))
            error_vel_down.append(abs(round(min(new_vel[i]) - vel_err[i],3)))
        
        #Lo ponemos así para que la funcion errorbar me lea los valores
        error_angulo = [error_ang_up,error_ang_down] 
        error_rad = [error_rad_up,error_rad_down]
        error_velocidad = [error_vel_up,error_vel_down]
        
        
        #Ahora sí, graficamos
        
        medida = round(m[n]*1000)
        plt.figure()
        plt.plot(tiempo1[0],angulo1[0], "--k", label="Promedio Experimental",  linewidth = 1.5, markersize = 3)        
        plt.errorbar(t_err,ang_err, yerr=error_angulo, fmt='.', color='k', label="Margen de error", alpha = 0.7)
        plt.plot(time2,angulo_grad2, "--r", linewidth = 2, markersize = 3, label="Simulación Teórica") #Esta es la única diferencia
        plt.title(r"Movimiento gradual: Comparativa $(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font)
        plt.ylabel(r"Ángulo $\theta° = {}$".format(an[n]), fontdict = font2)
        plt.xlabel("Tiempo ($s$)", fontdict=font2)
        plt.legend(loc = 'lower right')# , bbox_to_anchor=(0.9, 0.5, 0.5, 0.5))
        plt.grid(linestyle="--")
        plt.show()

        #Angulo en radianes
        plt.figure()
        plt.plot(tiempo1[0],angulo_rad1[0], "--k", label="Promedio Experimental", linewidth = 1.5, markersize = 3)
        plt.errorbar(t_err,ang_rad_err, yerr=error_rad, fmt='.', color='k', label="Margen de error", alpha = 0.7)
        plt.plot(time2,angulo2, "--r", linewidth = 2, markersize = 3, label = "Simulación") #Esta es la única diferencia
        plt.title(r"Movimiento Angular: Comparativa $(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font)
        plt.ylabel(r"Ángulo $\theta~(Rad)$", fontdict = font2)
        plt.xlabel("Tiempo ($s$)", fontdict=font2)
        plt.legend(loc = 'lower right') #, bbox_to_anchor=(0.9, 0.5, 0.5, 0.5))
        plt.grid(linestyle="--")
        plt.show()

        #Movimiento Angular
        plt.figure()
        plt.plot(tiempo1[0],velocidad1[0], "--k", label="Promedio Experimental", linewidth = 1.5, markersize = 3)
        plt.errorbar(t_err,vel_err, yerr=error_velocidad, fmt='.', color='k', label="Margen de error", alpha = 0.7)
        plt.plot(time2,w2, "--r", linewidth = 2, markersize = 3, label = "Simulación Teórica") #Esta es la única diferencia
        plt.title(r"Velocidad Angular: Comparativa $(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font)
        plt.ylabel(r"Velocidad angular $\theta~(Rad)$", fontdict = font2)
        plt.xlabel("Tiempo ($s$)", fontdict=font2)
        plt.legend(loc = "lower left") #, bbox_to_anchor=(0.9, 0.5, 0.5, 0.5))
        plt.grid(linestyle="--")
        plt.show()
   
    
    #Estimando la gravedad
    prom_periodo, contador = 0, 0
    for i in range(0,len(tiempo1)):
        pos = np.where(angulo1[i] == angulo1[i][0])   
        periodo=0
        if len(pos[0])>1:
            periodo = (tiempo1[i][pos[0][1]])-(tiempo1[i][pos[0][0]])
    #Hay algunos experimentos que no me dejan hallar el periodo pq no completan la oscilación
    #pero aún así podemos trabajar con estos, cuando el periodo está en ese rango (Normalmente vale 2.2 o 1.5 en casi todos):
        if periodo < 3 and periodo >1: 
            prom_periodo = prom_periodo + periodo
            gravedad1 = (4*math.pi**2 * L[n]) / (periodo**2)
            print("La gravedad del experimento {} es de aproximadamente:".format(i+1) , round(gravedad1,3))
            contador+= 1
        if contador >0:
            periodo_tot = prom_periodo/contador
        elif contador==0:
            periodo_tot = 2.1
    print("El periodo promedio es: ", round(periodo_tot,3))
    
    #Gravedad:
    gravedad = (4*math.pi**2 * L[n]) / (round(periodo_tot,3)**2)
    print("La gravedad es de aproximadamente:" , round(gravedad,3))
    contador2, prom_gravedad = 0, 0
    
        #Gravedad x2: Usando la fórmula mencionada en el informe

    suma=0
    for x in range(0,100):
        grado = an[n] * math.pi / 180
        a = ((math.factorial(2*x))/((2**x)*math.factorial(x))**2)**2
        b = a * math.sin(grado/2)**(2*x)
        suma += b
    sumatoria = suma**2
    gravity = 4*(math.pi**2) * L[n] * sumatoria / (round(periodo_tot,3))**2
    print("Cuando el ángulo inicial es de {} y la longitud de la cuerda de {}, la gravedad es de: {}".format(an[n],L[n], round(gravity,4)))
    
    

    #Estimando la diferencia de cuadrados:
    #La función del ángulo está dado por 0°+Cos(0.99pi*X) aprox, tons:
    def dispersion(n):
        medida = round(m[n]*1000)    
        forma = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        var_tot, contador3 = 0,0
        diferencia = []
        contador, varianza = 0, 0
        for i in range(0,len(angulo1[0])): 
            posi = (np.where(time == tiempo1[0][i])) #Encontrando los tiempos de tracker en el modelo teórico
            if len(posi[0]) == 1:   
                angle = angulo_grad[n][posi[0][0]] #El valor teórico para cada momento en tracker
                diferencia.append(angulo1[0][i] - angle)
                contador += 1
                varianza += (angulo1[0][i] - angle)**2

        new_dif = diferencia[:int((len(diferencia)/2))] #Tomamos los primeros valores, donde la fricción aún no ha afectado tanto
        new_time = tiempo1[0][:int((len(diferencia)/2))]
        
        plt.scatter(new_time,new_dif,  label="Promedio Experimental")
            
        
        plt.title(r"Análisis de residuos: $(\theta = {}°)$ - ${}$ml - ${}$m.".format(an[n],medida,L[n]), fontdict = font)
        plt.ylabel("Residuo", fontdict = font2)
        plt.xlabel("Tiempo ($s$)", fontdict=font2)
        plt.axhline(y = 0, linestyle = '--', color = 'black', lw=2)
        plt.legend(loc = "upper left")
        plt.grid(linestyle="--")
        plt.show()
        
    def dispersion_promedio(n):
        medida = round(m[n]*1000)    
        var_tot, contador3 = 0,0
        for j in range(1,len(angulo1)):
            diferencia = angulo1[0] - angulo1[j]    
            sumatoria = sum((angulo[0]-angulo[j])**2)
            contador = len(angulo[0])
            new_dif = diferencia[:int((len(diferencia)/2))] #Tomamos los primeros valores, donde la fricción aún no ha afectado tanto
            new_time = tiempo1[j][:int((len(diferencia)/2))]
            var = (sumatoria/contador)**(1/2)
            var_tot += var
            contador3 += 1
            print("La desviación cuadrática del experimento {} es de: {}".format((j),round(var,2)))
        varianza = var_tot/contador3
        print("En promedio, este caso presenta una desviación cuadrática de: {}".format(round(varianza,2)))
    dispersion(n)  
    grafica_experimento(n)
    grafica_comparativa(n, time)
    dispersion_promedio(n)    

##Ejecutando la función para cada caso (son 15), los ejecuto por aparte pq jupyter no me deja poner más de 20 gráficas en un bloque
obtencion_datos(0)
obtencion_datos(1)
obtencion_datos(2)
obtencion_datos(3)
obtencion_datos(4)
obtencion_datos(5)
obtencion_datos(6)
obtencion_datos(7)
obtencion_datos(8)
obtencion_datos(9)
obtencion_datos(10)
obtencion_datos(11)
obtencion_datos(12)
obtencion_datos(13)
obtencion_datos(14)
obtencion_datos(15)
