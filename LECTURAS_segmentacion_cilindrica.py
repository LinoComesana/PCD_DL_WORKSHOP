#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:02:04 2021

@author: lino
"""

import os
import open3d as o3d
import numpy as np


def lectura_segmentaciones(ruta):


    #ruta = '/home/lino/Documentos/TESTEO_ZEBGO_Nubes/NDP_zebRevo_xures_julio_21/prueba/arboles_manuales_troncos_eliminados/automatizacion/PLANO_TRONCOS_1.000000_0.250000_DBSCAN_TRONCOS_0.100000_1_DBSCAN_CILINDRO_0.900000_10_UMBRAL_0.026263'
    # /home/lino/Documentos/TESTEO_ZEBGO_Nubes/NDP_zebRevo_xures_julio_21/prueba/DBSCAN_TRONCOS_0.300000_1_DBSCAN_CILINDRO_0.800000_10_UMBRAL_0.250000
    
    os.chdir(ruta)
    
    SEGMENTOS_aux = []
    
    
    for i in range(len(os.listdir(os.getcwd()))):
        
        if os.listdir(os.getcwd())[i].endswith('.pcd'):
            if os.listdir(os.getcwd())[i].startswith('arbol'):
                SEGMENTOS_aux.append(o3d.io.read_point_cloud(os.listdir(os.getcwd())[i]))
            else:
                if os.listdir(os.getcwd())[i].endswith('.root'):
                    pass
                else:
                    resto = o3d.io.read_point_cloud(os.listdir(os.getcwd())[i])
    
    if len(SEGMENTOS_aux) != 0:
        
        SEGMENTOS = {}
        
        for i in range(len(SEGMENTOS_aux)):
            SEGMENTOS[i] = SEGMENTOS_aux[i]
            
        
        #pcd = o3d.io.read_point_cloud("../../test_data/fragment.pcd")
        
        # #-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
        # # PARÓN PARA VISUALIZAR:
        # visor.custom_draw_geometry_with_key_callback(SEGMENTOS,segmentado=True,
        #                                              pcd2=SEGMENTOS[0])
        # #-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
        
        # Lo necesito para hacer testeos rápidos con root tal y como lo tengo montado:
        FUSIONES = SEGMENTOS
    
    
    else:
        # Si estamos en un directorio donde las segmentaciones sólo están en forma-
        # to binario .root:
        import ROOT
        
        for archivo in os.listdir():
            if archivo.endswith('.root'):
                archivo_root = archivo
    
    
        # Para volver acceder a este Tree faríamos o seguinte:
        
        # Primeiro defino as listas onde almacenarei os valores que obteño do TTree
        # (Chamándoas co mesmo nome que a lectura con Python podo checkear se vai ben)
        coordenadas_x = []
        coordenadas_y = []
        coordenadas_z = [] 
        rojo = []
        verde = []
        azul = []
        
        # Abrimos o arquivo.root onde temos o TTree:
        ARQUIVO = ROOT.TFile(archivo_root,"update")
        
        aa = ARQUIVO.GetListOfKeys()
        Numero_de_arboles = aa.GetSize()
        
        SEGMENTOS = {}
        
        for A in range(Numero_de_arboles):
            # Xeramos un punteiro ao TTree que buscamos:
            tree_que_busco = ARQUIVO.Get("Arbol_%i"%A)
            
            # Almacenamos o número total de entradas que ten o TTree, é dicir,o número de
            # filas que conteñe se fixésemos a instrucción: tree_que_busco.Scan()
            Numero_de_entradas = tree_que_busco.GetEntries()
            
            puntos_arbol = []
            color_arbol = []
            
            # Agora iteramos en cada TBranch do Tree:
            for i in range(Numero_de_entradas):
                entry = tree_que_busco.GetEntry(i) # Metémonos na fila i-ésima
                puntos_arbol.append([tree_que_busco.coordenadas_x,
                                     tree_que_busco.coordenadas_y,
                                     tree_que_busco.coordenadas_z])
                #coordenadas_x.append(tree_que_busco.coordenadas_x)
                #coordenadas_y.append(tree_que_busco.coordenadas_y)
                #coordenadas_z.append(tree_que_busco.coordenadas_z)
                
                color_arbol.append([tree_que_busco.rojo,
                                    tree_que_busco.verde,
                                    tree_que_busco.azul])
                # rojo.append(tree_que_busco.rojo)
                # verde.append(tree_que_busco.verde)
                # azul.append(tree_que_busco.azul)
        
            arbol = o3d.geometry.PointCloud()
            arbol.points = o3d.utility.Vector3dVector(puntos_arbol)    
            # Es el mismo color para todos los puntos del árbol así que:
            arbol.paint_uniform_color(color_arbol[0])
        
            SEGMENTOS[A] = arbol
        
        # #-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
        # # PARÓN PARA VISUALIZAR:
        # visor.custom_draw_geometry_with_key_callback(SEGMENTOS,segmentado=True,
        #                                              pcd2=SEGMENTOS[0])
        # #-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
        
        # Lo necesito para hacer testeos rápidos con root tal y como lo tengo montado:
        FUSIONES = SEGMENTOS

    return SEGMENTOS






