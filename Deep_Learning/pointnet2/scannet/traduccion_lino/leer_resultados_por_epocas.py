#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:49:17 2022

@author: lino
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

RUTA_LOG_ABSOLUTA = '/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_autopista_5_capas_seccion_transversal_batch1_50k'

os.chdir(RUTA_LOG_ABSOLUTA)

# PRIMERO LEEMOS LOS PESOS DE LAS LABELS:


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
    
archivo = 'autopista_5_capas_seccion_transversal_batch1_50k_accuracies_y_losses_one_epoch.txt'

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













# Ahora cogemos la época con mejor accuracy y vemos qué predicción hizo:
    
import open3d as o3d

indice = np.argmax(ACCURACIES)

epoca_mejor_precision = epocas[indice]

with open("prediccion_nube_epoca_%i.npy"%epoca_mejor_precision, 'rb') as f:
    aug_data = np.load(f)[0]
    pred_val = np.load(f)[0]
        
indices_arboles = np.where(pred_val == 0)
indices_DTM = np.where(pred_val == 1)
indices_carretera = np.where(pred_val == 2)
indices_talud = np.where(pred_val == 3)
indices_senhal = np.where(pred_val == 4)
indices_barrera_quitamiedos_1 = np.where(pred_val == 5)
indices_barrera_quitamiedos_2 = np.where(pred_val == 6)
indices_arcen = np.where(pred_val == 7)
indices_mediana = np.where(pred_val == 8)
indices_barrera_jersey = np.where(pred_val == 9)
indices_berma = np.where(pred_val == 10)

colores = np.zeros(shape=aug_data.shape)
colores[indices_arboles] = [0,1,0]
colores[indices_DTM] = [1,0,0]
colores[indices_carretera] = [0,0,0]
colores[indices_talud] = [1,0.5,0.5]
colores[indices_senhal] = [0,0,1]
colores[indices_barrera_quitamiedos_1] = [0.4,0.4,0.4]
colores[indices_barrera_quitamiedos_2] = [0.4,0.4,0.4]
colores[indices_arcen] = [1,1,0]
colores[indices_mediana] = [0.1,0.21,0.1]
colores[indices_barrera_jersey] = [0.7,0.4,0.1]
colores[indices_berma] = [0.5,1,0.5]

nube_clasificada = o3d.geometry.PointCloud()
nube_clasificada.points = o3d.utility.Vector3dVector(aug_data)
nube_clasificada.colors = o3d.utility.Vector3dVector(colores)
o3d.visualization.draw(nube_clasificada)
o3d.io.write_point_cloud("nube_clasificada.pcd", nube_clasificada)






