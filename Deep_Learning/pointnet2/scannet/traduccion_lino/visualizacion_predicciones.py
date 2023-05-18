#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:59:08 2022

@author: lino
"""


# LECTURA DE PREDICCIONES DE SEGMENTACION CON POINTNET++:
    
import numpy as np
import open3d as o3d
import os


# Ruta al directorio /log donde están los resultados tanto de las fases de en-
# trenamiento/validación como de la predicción:
RUTA = '/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_autopista_5_capas_seccion_transversal'


os.chdir(RUTA)
with open("prediccion_nube_epoca_10.npy", 'rb') as f:
    aug_data = np.load(f)[0]
    pred_val = np.load(f)[0]
        
indices_arboles = np.where(pred_val == 0)
indices_DTM = np.where(pred_val == 1)

colores = np.zeros(shape=aug_data.shape)
colores[indices_arboles] = [0,1,0]
colores[indices_DTM] = [1,0,0]

nube_clasificada = o3d.geometry.PointCloud()
nube_clasificada.points = o3d.utility.Vector3dVector(aug_data)
nube_clasificada.colors = o3d.utility.Vector3dVector(colores)
o3d.visualization.draw(nube_clasificada)
