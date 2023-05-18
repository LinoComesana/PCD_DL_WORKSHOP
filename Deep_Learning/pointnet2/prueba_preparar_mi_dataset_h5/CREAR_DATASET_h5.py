#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:07:10 2022

@author: lino
"""

import h5py
import os
import numpy as np

def crear_dataset_h5(Ruta_nubes_artificiales):

    os.chdir(Ruta_nubes_artificiales)
    
    if 'set_entrenamiento.h5' not in os.listdir(os.getcwd()):
        
        Numero_nubes_artificiales = len(os.listdir(os.getcwd()))
        
        Numero_nubes_entrenamiento = int(0.8*Numero_nubes_artificiales)
        Numero_nubes_test = Numero_nubes_artificiales - Numero_nubes_entrenamiento
        
        for i in range(Numero_nubes_artificiales):
            
            if i < Numero_nubes_entrenamiento:
                
                datos_arboles = []
                datos_DTM = []
                
                # Nubes de entrenamiento
            
                os.chdir(Ruta_nubes_artificiales)
                
                os.chdir('Nube_artificial_%i'%i)
                
        
                for archivo in os.listdir(os.getcwd()):
                    if archivo.endswith('DTM.npy'):
                        puntos_DTM_aux = np.load(archivo)
                        datos_DTM.append(puntos_DTM_aux)
                    if archivo.endswith('arboles.npy'):
                        puntos_arboles_aux = np.load(archivo)
                        datos_arboles.append(puntos_arboles_aux)
                    
                if i == Numero_nubes_entrenamiento - 1:
                    # Estamos en la última nube del set de entrenamiento, monta-
                    # mos el h5file:
                        
                     datos_DTM = np.array(datos_DTM)
                     datos_arboles = np.array(datos_arboles)
                        
                     archivo_f5 = h5py.File('set_entrenamiento.h5')
                 
                     data = np.vstack((puntos_arboles,puntos_DTM))
                     
                     # Hago que los árboles tengan label 1 mientras que el DTM sea 0:
                     labels_arboles = np.full((len(puntos_arboles),1),1)
                     labels_DTM = np.full((len(puntos_DTM),1),0)
                     
                     label = np.vstack((labels_arboles,labels_DTM))
                
                
                
                
                
                '''
                if i == 0:
                    archivo_f5 = h5py.File('set_entrenamiento.h5')
                
                    data = np.vstack((puntos_arboles,puntos_DTM))
                    
                    # Hago que los árboles tengan label 1 mientras que el DTM sea 0:
                    labels_arboles = np.full((len(puntos_arboles),1),1)
                    labels_DTM = np.full((len(puntos_DTM),1),0)
                    
                    label = np.vstack((labels_arboles,labels_DTM))
                
                
                else:
                    
                    data = np
                '''
                
                
            else:
                # Nubes de testeo
            
                os.chdir(Ruta_nubes_artificiales)
                
                os.chdir('Nube_artificial_%i'%i)
            
                for archivo in os.listdir(os.getcwd()):
                    if archivo.endswith('DTM.npy'):
                        puntos_DTM_aux = np.load(archivo)
                    if archivo.endswith('arboles.npy'):
                        puntos_arboles_aux = np.load(archivo)
            
            
            
            
            
    archivo_f5.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype='float32',
    )
    
    
    archivo_f5.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=4,
            dtype='uint8',
    )
            
    
    archivo_f5.close()
