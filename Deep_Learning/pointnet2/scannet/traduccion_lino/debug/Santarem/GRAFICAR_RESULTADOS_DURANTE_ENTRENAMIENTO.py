#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:27:20 2022

@author: lino
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import time

NUMERO_DE_EPOCAS_GLOBAL = 10

for EPOCA in range(NUMERO_DE_EPOCAS_GLOBAL):

    RUTA_LOG_ABSOLUTA = '/home/lino/Documentos/PCD_DL_WORKSHOP/PCD_DL_WORKSHOP/Deep_Learning/pointnet2/log_WORKSHOP_VIERNES'
    
    os.chdir(RUTA_LOG_ABSOLUTA)
    
    # PRIMERO LEEMOS LOS PESOS DE LAS LABELS:
    
    while 'label_weights_epoca_%i.npy'%(EPOCA) not in os.listdir():
        print('Esperamos a que se genere el archivo...(60 s)')
        time.sleep(60)
    
    
    Lista_auxiliar = []
    for archivo in os.listdir():
        if archivo.startswith('label_weights_epoca'):
            archivo = archivo[20:].replace('.npy','')
            archivo = int(archivo)
            Lista_auxiliar.append(archivo)
    
    Lista_auxiliar = np.array(Lista_auxiliar)
    Lista_auxiliar.sort()
    
    NUMERO_DE_EPOCAS = np.max(Lista_auxiliar) + (Lista_auxiliar[1]-(Lista_auxiliar[0]))
    
    epocas = np.copy(Lista_auxiliar)
    del Lista_auxiliar
    
    Lista_labelweights = []
    for contador_epoca in range(NUMERO_DE_EPOCAS):
        try:
            with open('label_weights_epoca_%i.npy'%contador_epoca, 'rb') as f:
                labelweights = np.load(f)
                # print(labelweights)
            Lista_labelweights.append(labelweights)
        except FileNotFoundError:
            pass
    
    Lista_labelweights = np.array(Lista_labelweights)
    
    plt.figure('RESULTADOS',figsize=(15,6))
    
    plt.subplot(3,1,1) # Filas: 3, columnas: 1, índice: 1
    
    for i in range(len(Lista_labelweights[0])-1):
        plt.plot(epocas,Lista_labelweights.take(i,1),label='Clase %i'%i)
    plt.legend()
    plt.xlabel('Épocas')
    plt.ylabel('Labelweights')
    
    
    
    # AHORA LEEMOS LAS ACCURACIES Y LOSSES:
        
    archivo = 'WORKSHOP_VIERNES_accuracies_y_losses_one_epoch.txt'
    
    while archivo not in os.listdir():
        time.sleep(60)
        
    ACCURACIES = []
    LOSSES = []
    
    with open(archivo) as f:
        lineas = f.readlines()
        for i in range(1,len(lineas)):
            if i == 1:
                epoca_leida = epocas[0]
                linea = lineas[i].split('    ')
                linea = np.array(linea[0:3]).astype(np.float64)
                Lista_auxiliar = []
                Lista_auxiliar.append(np.array([float(linea[0]),float(linea[1])]))
            else:
                linea = lineas[i].split('    ')
                linea = np.array(linea[0:3]).astype(np.float64)
                if int(linea[2]) == epoca_leida:
                    Lista_auxiliar.append(np.array([float(linea[0]),float(linea[1])]))
                else:
                    indice_accuracy_maxima = np.argmax(np.max(Lista_auxiliar,axis=1))
                    ACCURACIES.append(Lista_auxiliar[indice_accuracy_maxima][0])
                    LOSSES.append(Lista_auxiliar[indice_accuracy_maxima][1])
                    epoca_leida = int(linea[2])
                    Lista_auxiliar = []
                    Lista_auxiliar.append(np.array([float(linea[0]),float(linea[1])]))
                    
            
    ACCURACIES = np.array(ACCURACIES)
    LOSSES = np.array(LOSSES)
    
    plt.subplot(3,1,2) # Filas: 1, columnas: 2, índice: 2
    plt.plot(epocas[0:-1],ACCURACIES,color='navy',label='Accuracies')
    plt.plot(epocas[0:-1],LOSSES,color='red',label='Losses')
    plt.legend()
    plt.xlabel('Épocas')
    
    plt.subplot(3,1,3) # Filas: 1, columnas: 2, índice: 3
    plt.plot(epocas[0:-1],ACCURACIES,color='navy')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracies')
    
    plt.savefig('RESULTADOS')
    plt.close('RESULTADOS')
    
    



