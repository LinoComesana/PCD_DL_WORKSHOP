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
RUTA = '/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_prueba_julio_2_ancho_celda_5'


os.chdir(RUTA)
with open("prediccion_nube.npy", 'rb') as f:
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
# Los saco de aquí: https://sashamaps.net/docs/resources/20-colors/ (buen con-
# traste entre ellos)
colores[indices_arboles] = np.array([60,180,75])/255.
colores[indices_DTM] = np.array([170,110,40])/255.
colores[indices_carretera] = np.array([0,0,0])/255.
colores[indices_talud] = np.array([245,130,48])/255.
colores[indices_senhal] = np.array([0,0,128])/255.
colores[indices_barrera_quitamiedos_1] = np.array([128,128,128])/255.
colores[indices_barrera_quitamiedos_2] = np.array([128,128,128])/255.
colores[indices_arcen] = np.array([255,255,25])/255.
colores[indices_mediana] = np.array([170,255,195])/255.
colores[indices_barrera_jersey] = np.array([145,30,180])/255.
colores[indices_berma] = np.array([250,190,212])/255.

nube_clasificada = o3d.geometry.PointCloud()
nube_clasificada.points = o3d.utility.Vector3dVector(aug_data)
nube_clasificada.colors = o3d.utility.Vector3dVector(colores)
o3d.visualization.draw(nube_clasificada)
o3d.io.write_point_cloud("nube_clasificada.pcd", nube_clasificada)
    
    
