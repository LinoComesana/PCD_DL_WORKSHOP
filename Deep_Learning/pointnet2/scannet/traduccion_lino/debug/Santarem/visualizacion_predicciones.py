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
# RUTA = '/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_PAPER_nacional_Santarem__def_4_layers_DEFAULT-PN++'
RUTA = '/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_PAPERRRRRRRRRRRR_BOSQUE_XURES'

os.chdir(RUTA)

epoca_seleccionada = 82

# épocas 6 y 14
with open("PREDICCION_NUBE_REAL_epoca_%i.npy"%epoca_seleccionada, 'rb') as f:
# with open("prediccion_nube_epoca_0.npy", 'rb') as f:
    aug_data = np.load(f)[0]
    pred_val = np.load(f)[0]
        
Santarem = True
    
    
if not Santarem:
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

else:
    indices_arboles = np.where(pred_val == 0)
    indices_DTM = np.where(pred_val == 1)
    indices_senhal = np.where(pred_val == 2)
    indices_barreras_quitamiedos = np.where(pred_val == 3)
    indices_mediana = np.where(pred_val == 4)
    indices_berma = np.where(pred_val == 5)
    indices_puntos_circulacion = np.where(pred_val == 6)
    
    colores = np.zeros(shape=aug_data.shape)
    # Los saco de aquí: https://sashamaps.net/docs/resources/20-colors/ (buen con-
    # traste entre ellos)
    colores[indices_arboles] = np.array([60,180,75])/255.
    colores[indices_DTM] = np.array([170,110,40])/255.
    colores[indices_senhal] = np.array([0,0,128])/255.
    colores[indices_barreras_quitamiedos] = np.array([128,128,128])/255.
    colores[indices_mediana] = np.array([170,255,195])/255.
    colores[indices_berma] = np.array([250,190,212])/255.
    colores[indices_puntos_circulacion] = np.array([0,0,0])/255.









# RESULTADOS VEGETACION EN AUTOPISTA PAPER:

nube_clasificada = o3d.geometry.PointCloud()
nube_clasificada.points = o3d.utility.Vector3dVector(aug_data)
nube_clasificada.colors = o3d.utility.Vector3dVector(colores)

# # nube_clasificada.paint_uniform_color([0,0,0])

# nube_entrenamiento = '/home/lino/Documentos/programas_pruebas_varias/segmentacion_python/segmentacion_bosques/aumentacion_de_datos/Nubes_artificiales_generadas/nubes_buenas/Nubes_artificiales_bosques/TRAIN/Nube_artificial_1/Nube_artificial_1.pcd'
# nube_ENTRENAMIENTO = o3d.io.read_point_cloud(nube_entrenamiento)
# # o3d.visualization.draw(nube_clasificada+nube_ENTRENAMIENTO)
o3d.visualization.draw(nube_clasificada)
o3d.io.write_point_cloud("nube_clasificada.pcd", nube_clasificada)
    


# with open("INDICES_PUNTOS_SELECCIONADOS_epoca_%i.npy"%epoca_seleccionada, 'rb') as f:
# # with open("prediccion_nube_epoca_0.npy", 'rb') as f:
#     indices_puntos_seleccionados = np.load(f)[0]


# ruta_paper_figuras = '/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/paper_figuras/autopista'


# import shutil
# shutil.copyfile(RUTA+"/INDICES_PUNTOS_SELECCIONADOS_epoca_%i.npy"%epoca_seleccionada, ruta_paper_figuras+'/INDICES_PUNTOS_SELECCIONADOS_epoca_%i.npy'%epoca_seleccionada)


# os.chdir(ruta_paper_figuras)
# o3d.io.write_point_cloud("entorno_clasificado.pcd", nube_clasificada)


