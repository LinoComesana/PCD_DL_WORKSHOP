import pickle
import os
import sys
import numpy as np
import pc_util
import scene_util
#import laspy as lp






#######################################################################
# OJOOOOOOOOOOOOOOOOOOOOOOOO
# Definimos el valor del ancho de la celda en metros. Recordemos que lo
# que hace Pointnet++ es escoger un punto aleatorio de la nube y trazar
# una celda cubica a su alrededor. Esa es la sample con la que trabaja:
# ancho_de_celda = 5
ancho_de_celda = 40
#######################################################################


def leer_puntos_clases_Santarem_txt(ruta_carpeta_con_txt):
    ruta_antes = os.getcwd()
    os.chdir(ruta_carpeta_con_txt)
    
    lista_txt_pa_leer = []
    for archivo in os.listdir(os.getcwd()):
        if archivo.endswith('.txt'):
            lista_txt_pa_leer.append(archivo)
            
    X = []
    Y = []
    Z = []
            
    for archivo_txt in lista_txt_pa_leer:
        with open(archivo_txt,mode='r') as f:
            lineas = f.readlines()
            for l in range(1,len(lineas)): # La primera linea es la caecera:
                linea = lineas[l]
                linea = linea.split(' ')
                X.append(float(linea[0]))
                Y.append(float(linea[1]))
                Z.append(float(linea[2]))
                
    os.chdir(ruta_antes)
    puntos_esta_categoria = np.stack((X,Y,Z)).T
    return puntos_esta_categoria
    



def leer_puntos_clases_Santarem_npy(ruta):
    ruta_antes = os.getcwd()
    os.chdir(ruta)
    
    for subarchivo in os.listdir(os.getcwd()):                        
        if subarchivo.endswith('DTM.npy'):
            puntos_DTM = np.load(subarchivo)
        if subarchivo.endswith('carretera.npy'):
            puntos_carretera = np.load(subarchivo)
        if subarchivo.endswith('barreras_quitamiedos.npy'):
            puntos_barreras_quitamiedos = np.load(subarchivo)
        if subarchivo.endswith('senhales.npy'):
            puntos_senhales = np.load(subarchivo)


    puntos_via_circulacion = puntos_carretera
    
    
    # Ya estan leidos todas las clases.
    # Voy a crear unos arrays de etiquetas:
        
    etiquetas_DTM = np.full(shape=(1,len(puntos_DTM)),fill_value=1)
    etiquetas_senhales = np.full(shape=(1,len(puntos_senhales)),fill_value=2)
    etiquetas_barreras_quitamiedos = np.full(shape=(1,len(puntos_barreras_quitamiedos)),fill_value=3)
    etiquetas_via_circulacion = np.full(shape=(1,len(puntos_via_circulacion)),fill_value=6)
                                
    # Creamos ahora el array de todos los puntos de la nube y 
    # el array de todas las labels semanticas de la nube:
        
    puntos_nube = np.vstack((puntos_DTM,
                             puntos_senhales,
                             puntos_barreras_quitamiedos,
                             puntos_via_circulacion))
    
    labels_semanticas_nube = np.vstack((etiquetas_DTM.T,
                                        etiquetas_senhales.T,
                                        etiquetas_barreras_quitamiedos.T,
                                        etiquetas_via_circulacion.T)).T[0]
    
    # Vamos a desordenar ambos arrays del mismo modo:
    puntos_nube,labels_semanticas_nube = desordenar_dos_arrays(puntos_nube,labels_semanticas_nube)
    
    # print('holahola')
    # print(puntos_nube.shape)
    
    
    # Anhadimos los puntos y sus labels al dataset que estamos
    # montando:
        
    Lista_puntos = []
    Lista_labels_semanticas = []
        
    Lista_puntos.append(puntos_nube)
    Lista_labels_semanticas.append(labels_semanticas_nube)
    
    # Volvemos a la carpeta original
    os.chdir(ruta_antes)
    

    return Lista_puntos,Lista_labels_semanticas



        
            

def desordenar_dos_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



class ScannetDataset():
        
    def __init__(self, root, npoints=8192, split='train',LINO=True,NUM_CLASSES = 2,downsampleamos = False, contador_dataset = 0, tamanho_batch = 1,tipo_de_nubes = '',archivo='',algunas_clases_fusionadas=''):
        self.NUM_CLASSES = NUM_CLASSES
        self.npoints = npoints
        self.root = root
        self.split = split
        self.tamanho_batch = tamanho_batch
        self.contador_dataset = contador_dataset
        self.tipo_de_nubes = tipo_de_nubes
        
        if not LINO:
            self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
            with open(self.data_filename,'rb') as fp:
                self.scene_points_list = pickle.load(fp)
                self.semantic_labels_list = pickle.load(fp)
        else:
            
            
            if split != 'prediccion':
            
                if algunas_clases_fusionadas:
                    # Consideramos taludes y DTM como la misma clase y lo mismo
                    # para arcenes y carretera:
                    
                    Lista_puntos = []
                    Lista_labels_semanticas = []
                    
                    # Voy a hacer que recorra todas las subcarpetas que haya en un di-
                    # rectorio TRAIN (O TEST) y asi recopile todas las nubes en una
                    # unica lista:
                    
                    os.chdir(root)
                    
                    # Estamos en la carpeta TRAIN (TEST), recorremos las subcarpetas:
                    
                    for a in range(contador_dataset,contador_dataset+tamanho_batch):
                        
                        archivo = 'Nube_artificial_%i'%a
                        
                        if archivo in os.listdir(os.getcwd()):
                            
                            # print('holaaaaaaaaaaaa\n')
                            # print(os.getcwd())
                            # print('holaaaaaaaaaaaa\n')
                            
                            os.chdir(os.getcwd()+'/'+archivo+'/numpy_arrays')
                             
                            
                            if 'carretera.npy' not in os.listdir(os.getcwd()):
                                nube_tipo_bosque = True
                            
                            
                            if not downsampleamos:
                                # Leemos sin hacer un downsampleo previo:
                                
                                print(os.getcwd())
                                
                                for subarchivo in os.listdir(os.getcwd()):                        
                                    if subarchivo.endswith('arboles.npy'):
                                        puntos_arboles = np.load(subarchivo)
                                    if subarchivo.endswith('DTM.npy'):
                                        puntos_DTM = np.load(subarchivo)
                                    if subarchivo.endswith('carretera.npy'):
                                        puntos_carretera = np.load(subarchivo)
                                    if subarchivo.endswith('talud.npy'):
                                        puntos_talud = np.load(subarchivo)
                                    if subarchivo.endswith('arcen.npy'):
                                        puntos_arcen = np.load(subarchivo)
                                    if subarchivo.endswith('barreras_quitamiedos.npy'):
                                        puntos_barreras_quitamiedos = np.load(subarchivo)
                                    if subarchivo.endswith('mediana.npy'):
                                        puntos_mediana = np.load(subarchivo)
                                    if subarchivo.endswith('senhales.npy'):
                                        puntos_senhales = np.load(subarchivo)
                                    if subarchivo.endswith('berma.npy'):
                                        puntos_berma = np.load(subarchivo)
                                    
                                    
                                    
                            
                                    
                            if downsampleamos:
                                # Hacemos un downsampleo previo:
                                for subarchivo in os.listdir(os.getcwd()):                        
                                    if subarchivo.endswith('arboles.npy'):
                                        puntos_arboles = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_arboles)), 100000 , replace=False)
                                        puntos_arboles = puntos_arboles[indices_seleccionados]
                                    if subarchivo.endswith('DTM.npy'):
                                        puntos_DTM = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_DTM)), 100000 , replace=False)
                                        puntos_DTM = puntos_DTM[indices_seleccionados]
                                    if subarchivo.endswith('carretera.npy'):
                                        puntos_carretera = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_carretera)), 100000 , replace=False)
                                        puntos_carretera = puntos_carretera[indices_seleccionados]
                                    if subarchivo.endswith('talud.npy'):
                                        puntos_talud = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_talud)), 100000 , replace=False)
                                        puntos_talud = puntos_talud[indices_seleccionados]
                                    if subarchivo.endswith('arcen.npy'):
                                        puntos_arcen = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_arcen)), 100000 , replace=False)
                                        puntos_arcen = puntos_arcen[indices_seleccionados]
                                    if subarchivo.endswith('barreras_quitamiedos.npy'):
                                        puntos_barreras_quitamiedos = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_barreras_quitamiedos)), 100000 , replace=False)
                                        puntos_barreras_quitamiedos = puntos_barreras_quitamiedos[indices_seleccionados]
                                    if subarchivo.endswith('mediana.npy'):
                                        puntos_mediana = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_mediana)), 100000 , replace=False)
                                        puntos_mediana = puntos_mediana[indices_seleccionados]
                                    if subarchivo.endswith('senhales.npy'):
                                        puntos_senhales = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_senhales)), 100000 , replace=False)
                                        puntos_senhales = puntos_senhales[indices_seleccionados]
                                    if subarchivo.endswith('berma.npy'):
                                        puntos_berma = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_berma)), 100000 , replace=False)
                                        puntos_berma = puntos_berma[indices_seleccionados]
                            
                            
                            # Voy a juntar los puntos de los arcenes y carretera
                            # como si fuesen la misma clase:
                            
                            try:
                                puntos_via_circulacion = np.concatenate((puntos_carretera,
                                                                     puntos_arcen))
                            except:
                                pass
                            
                            # Lo mismo para DTM y taludes:
                            
                            if not nube_tipo_bosque:
                                puntos_DTM = np.concatenate((puntos_DTM,puntos_talud))
                            
                            # Ya estan leidos todas las clases.
                            # Voy a crear unos arrays de etiquetas:
                                
                                
                            carretera_autopista = True # Lo cambio si se detec-
                            # ta una mediana
                            carretera_nacional = False
                            carretera_local = False
                            
                            etiquetas_arboles = np.full(shape=(1,len(puntos_arboles)),fill_value=0)
                            etiquetas_DTM = np.full(shape=(1,len(puntos_DTM)),fill_value=1)
                          
                            try:
                                etiquetas_senhales = np.full(shape=(1,len(puntos_senhales)),fill_value=2)
                            except:
                                pass
                            try:
                                etiquetas_barreras_quitamiedos = np.full(shape=(1,len(puntos_barreras_quitamiedos)),fill_value=3)
                            except:
                                carretera_local = True
                            try:
                                etiquetas_mediana = np.full(shape=(1,len(puntos_mediana)),fill_value=4)
                            except:
                                carretera_autopista = False
                                if carretera_local == False:
                                    carretera_nacional = True
                            try:
                                etiquetas_berma = np.full(shape=(1,len(puntos_berma)),fill_value=5)
                            except:
                                pass
                            
                            try:
                                etiquetas_via_circulacion = np.full(shape=(1,len(puntos_via_circulacion)),fill_value=6)
                            except:
                                pass
                            # Creamos ahora el array de todos los puntos de la nube y 
                            # el array de todas las labels semanticas de la nube:
                            
                               
                            
                            if not nube_tipo_bosque:
                            
                                if carretera_autopista:
                                    
                                    
                                    puntos_nube = np.vstack((puntos_arboles,
                                                             puntos_DTM,
                                                             puntos_senhales,
                                                             puntos_barreras_quitamiedos,
                                                             puntos_mediana,
                                                             puntos_berma,
                                                             puntos_via_circulacion))
                                    
                                    labels_semanticas_nube = np.vstack((etiquetas_arboles.T,
                                                                        etiquetas_DTM.T,
                                                                        etiquetas_senhales.T,
                                                                        etiquetas_barreras_quitamiedos.T,
                                                                        etiquetas_mediana.T,
                                                                        etiquetas_berma.T,
                                                                        etiquetas_via_circulacion.T)).T[0]
                                
                                
                                if carretera_nacional:
                                    
                                    puntos_nube = np.vstack((puntos_arboles,
                                                             puntos_DTM,
                                                             puntos_senhales,
                                                             puntos_barreras_quitamiedos,
                                                             puntos_berma,
                                                             puntos_via_circulacion))
                                    
                                    labels_semanticas_nube = np.vstack((etiquetas_arboles.T,
                                                                        etiquetas_DTM.T,
                                                                        etiquetas_senhales.T,
                                                                        etiquetas_barreras_quitamiedos.T,
                                                                        etiquetas_berma.T,
                                                                        etiquetas_via_circulacion.T)).T[0]
                                    
                                    
                                    
                                if carretera_local:
                                    
                                    
                                    puntos_nube = np.vstack((puntos_arboles,
                                                             puntos_DTM,
                                                             puntos_senhales,
                                                             puntos_via_circulacion))
                                    
                                    labels_semanticas_nube = np.vstack((etiquetas_arboles.T,
                                                                        etiquetas_DTM.T,
                                                                        etiquetas_senhales.T,
                                                                        etiquetas_via_circulacion.T)).T[0]
                                    
                            if nube_tipo_bosque:
                                
                                    puntos_nube = np.vstack((puntos_arboles,
                                                             puntos_DTM))
                                    
                                    labels_semanticas_nube = np.vstack((etiquetas_arboles.T,
                                                                        etiquetas_DTM.T)).T[0]
                            
                            
                            
                            
                            # Vamos a desordenar ambos arrays del mismo modo:
                            puntos_nube,labels_semanticas_nube = desordenar_dos_arrays(puntos_nube,labels_semanticas_nube)
                            
                            # print('holahola')
                            # print(puntos_nube.shape)
                            
                            
                            # Anhadimos los puntos y sus labels al dataset que estamos
                            # montando:
                                
                            Lista_puntos.append(puntos_nube)
                            Lista_labels_semanticas.append(labels_semanticas_nube)
                            
                            # Volvemos a la carpeta TRAIN (TEST)
                            os.chdir(root)
                    
                    
                else:
                # Leemos el dataset de forma normal:
            
                    Lista_puntos = []
                    Lista_labels_semanticas = []
                    
                    # Voy a hacer que recorra todas las subcarpetas que haya en un di-
                    # rectorio TRAIN (O TEST) y asi recopile todas las nubes en una
                    # unica lista:
                    
                    os.chdir(root)
                    
                    # Estamos en la carpeta TRAIN (TEST), recorremos las subcarpetas:
                    
                    for a in range(contador_dataset,contador_dataset+tamanho_batch):
                        
                        archivo = 'Nube_artificial_%i'%a
                        
                        if archivo in os.listdir(os.getcwd()):
                            
                            os.chdir(archivo+'/numpy_arrays')
                                                    
                            if not downsampleamos:
                                # Leemos sin hacer un downsampleo previo:
                                for subarchivo in os.listdir(os.getcwd()):                        
                                    if subarchivo.endswith('arboles.npy'):
                                        puntos_arboles = np.load(subarchivo)
                                    if subarchivo.endswith('DTM.npy'):
                                        puntos_DTM = np.load(subarchivo)
                                    if subarchivo.endswith('carretera.npy'):
                                        puntos_carretera = np.load(subarchivo)
                                    if subarchivo.endswith('talud.npy'):
                                        puntos_talud = np.load(subarchivo)
                                    if subarchivo.endswith('arcen.npy'):
                                        puntos_arcen = np.load(subarchivo)
                                    if subarchivo.endswith('barreras_quitamiedos.npy'):
                                        puntos_barreras_quitamiedos = np.load(subarchivo)
                                    if subarchivo.endswith('mediana.npy'):
                                        puntos_mediana = np.load(subarchivo)
                                    if subarchivo.endswith('senhales.npy'):
                                        puntos_senhales = np.load(subarchivo)
                                    if subarchivo.endswith('berma.npy'):
                                        puntos_berma = np.load(subarchivo)
                                    
                                    
                            if downsampleamos:
                                # Hacemos un downsampleo previo:
                                for subarchivo in os.listdir(os.getcwd()):                        
                                    if subarchivo.endswith('arboles.npy'):
                                        puntos_arboles = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_arboles)), 100000 , replace=False)
                                        puntos_arboles = puntos_arboles[indices_seleccionados]
                                    if subarchivo.endswith('DTM.npy'):
                                        puntos_DTM = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_DTM)), 100000 , replace=False)
                                        puntos_DTM = puntos_DTM[indices_seleccionados]
                                    if subarchivo.endswith('carretera.npy'):
                                        puntos_carretera = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_carretera)), 100000 , replace=False)
                                        puntos_carretera = puntos_carretera[indices_seleccionados]
                                    if subarchivo.endswith('talud.npy'):
                                        puntos_talud = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_talud)), 100000 , replace=False)
                                        puntos_talud = puntos_talud[indices_seleccionados]
                                    if subarchivo.endswith('arcen.npy'):
                                        puntos_arcen = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_arcen)), 100000 , replace=False)
                                        puntos_arcen = puntos_arcen[indices_seleccionados]
                                    if subarchivo.endswith('barreras_quitamiedos.npy'):
                                        puntos_barreras_quitamiedos = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_barreras_quitamiedos)), 100000 , replace=False)
                                        puntos_barreras_quitamiedos = puntos_barreras_quitamiedos[indices_seleccionados]
                                    if subarchivo.endswith('mediana.npy'):
                                        puntos_mediana = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_mediana)), 100000 , replace=False)
                                        puntos_mediana = puntos_mediana[indices_seleccionados]
                                    if subarchivo.endswith('senhales.npy'):
                                        puntos_senhales = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_senhales)), 100000 , replace=False)
                                        puntos_senhales = puntos_senhales[indices_seleccionados]
                                    if subarchivo.endswith('berma.npy'):
                                        puntos_berma = np.load(subarchivo)
                                        indices_seleccionados = np.random.choice(range(len(puntos_berma)), 100000 , replace=False)
                                        puntos_berma = puntos_berma[indices_seleccionados]
                            
                                        
                            
                            # Ya estan leidos todas las clases.
                            # Voy a crear unos arrays de etiquetas:
                                
                            etiquetas_arboles = np.full(shape=(1,len(puntos_arboles)),fill_value=0)
                            etiquetas_DTM = np.full(shape=(1,len(puntos_DTM)),fill_value=1)
                            etiquetas_carretera = np.full(shape=(1,len(puntos_carretera)),fill_value=2)
                            etiquetas_talud = np.full(shape=(1,len(puntos_talud)),fill_value=3)
                            etiquetas_senhales = np.full(shape=(1,len(puntos_senhales)),fill_value=4)
                            etiquetas_barreras_quitamiedos = np.full(shape=(1,len(puntos_barreras_quitamiedos)),fill_value=5)
                            etiquetas_arcen = np.full(shape=(1,len(puntos_arcen)),fill_value=6)
                            etiquetas_mediana = np.full(shape=(1,len(puntos_mediana)),fill_value=7)
                            etiquetas_berma = np.full(shape=(1,len(puntos_berma)),fill_value=8)
                            
                            # Creamos ahora el array de todos los puntos de la nube y 
                            # el array de todas las labels semanticas de la nube:
                                
                            puntos_nube = np.vstack((puntos_arboles,
                                                     puntos_DTM,
                                                     puntos_carretera,
                                                     puntos_talud,
                                                     puntos_senhales,
                                                     puntos_barreras_quitamiedos,
                                                     puntos_arcen,
                                                     puntos_mediana,
                                                     puntos_berma))
                            
                            labels_semanticas_nube = np.vstack((etiquetas_arboles.T,
                                                                etiquetas_DTM.T,
                                                                etiquetas_carretera.T,
                                                                etiquetas_talud.T,
                                                                etiquetas_senhales.T,
                                                                etiquetas_barreras_quitamiedos.T,
                                                                etiquetas_arcen.T,
                                                                etiquetas_mediana.T,
                                                                etiquetas_berma.T)).T[0]
                            
                            # Vamos a desordenar ambos arrays del mismo modo:
                            puntos_nube,labels_semanticas_nube = desordenar_dos_arrays(puntos_nube,labels_semanticas_nube)
                            
                            # print('holahola')
                            # print(puntos_nube.shape)
                            
                            
                            # Anhadimos los puntos y sus labels al dataset que estamos
                            # montando:
                                
                            Lista_puntos.append(puntos_nube)
                            Lista_labels_semanticas.append(labels_semanticas_nube)
                            
                            # Volvemos a la carpeta TRAIN (TEST)
                            os.chdir(root)
                            
                    # ###################################################################
                    # # PARON PARA VISUALIZAR:
                    # import open3d as o3d
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(Lista_puntos[0])
                    # o3d.visualization.draw(pcd)                    
                    # ###################################################################                    
                    
                # Adecuo las variables que usan los desarrolladores de PointNet a 
                # las mias:
                
                self.scene_points_list = Lista_puntos
                self.semantic_labels_list = Lista_labels_semanticas
                
                # if split == 'test':
                #     print('hola')
                #     print(Lista_puntos)
            
            
            
            
            
            
            
            
        if split=='train':
            # labelweights = np.zeros(21)
            labelweights = np.zeros(NUM_CLASSES)
            for seg in self.semantic_labels_list:
                # tmp,_ = np.histogram(seg,range(22))
                tmp,_ = np.histogram(seg,range(NUM_CLASSES+1))
            labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            # self.labelweights = np.ones(21)
            self.labelweights = np.ones(NUM_CLASSES)
        elif split=='prediccion':
            
            # En el caso de hacer predicciones sobre la nube etiquetada a mano:
            # if root == '/home/lino/Documentos/NDP_carreteras_Santarem_Dani/nubes_recortadas/nube_real_etiquetada_a_mano':
            if root == '/home/lino/Documentos/programas_pruebas_varias/segmentacion_python/segmentacion_bosques/aumentacion_de_datos/Nubes_artificiales_generadas/nubes_buenas/dataset_carretera_nacional/PREDICT':
                os.chdir(root)
            
                Lista_puntos = []
                Lista_labels_semanticas = []
                
                nubes_en_formato_txt = False
                for archivo in os.listdir(os.getcwd()):
                    if archivo.endswith('.txt'):
                        nubes_en_formato_txt = True
                        break
                
                if nubes_en_formato_txt:
                
                    puntos_arboles = leer_puntos_clases_Santarem_txt(root+'/arboles')
                    puntos_DTM = leer_puntos_clases_Santarem_txt(root+'/DTM')
                    puntos_senhales = leer_puntos_clases_Santarem_txt(root+'/senhales')
                    puntos_barreras_quitamiedos = leer_puntos_clases_Santarem_txt(root+'/quitamiedos')
                    puntos_mediana = leer_puntos_clases_Santarem_txt(root+'/mediana')
                    puntos_berma = leer_puntos_clases_Santarem_txt(root+'/berma')
                    puntos_via_circulacion = leer_puntos_clases_Santarem_txt(root+'/puntos_circulacion')
                
                    etiquetas_arboles = np.full(shape=(1,len(puntos_arboles)),fill_value=0)
                    etiquetas_DTM = np.full(shape=(1,len(puntos_DTM)),fill_value=1)
                    etiquetas_senhales = np.full(shape=(1,len(puntos_senhales)),fill_value=2)
                    etiquetas_barreras_quitamiedos = np.full(shape=(1,len(puntos_barreras_quitamiedos)),fill_value=3)
                    etiquetas_mediana = np.full(shape=(1,len(puntos_mediana)),fill_value=4)
                    etiquetas_berma = np.full(shape=(1,len(puntos_berma)),fill_value=5)
                    etiquetas_via_circulacion = np.full(shape=(1,len(puntos_via_circulacion)),fill_value=6)
                
                    puntos_nube = np.vstack((puntos_arboles,
                                             puntos_DTM,
                                             puntos_senhales,
                                             puntos_barreras_quitamiedos,
                                             puntos_mediana,
                                             puntos_berma,
                                             puntos_via_circulacion))
                    
                    labels_semanticas_nube = np.vstack((etiquetas_arboles.T,
                                                        etiquetas_DTM.T,
                                                        etiquetas_senhales.T,
                                                        etiquetas_barreras_quitamiedos.T,
                                                        etiquetas_mediana.T,
                                                        etiquetas_berma.T,
                                                        etiquetas_via_circulacion.T)).T[0]
            
                if not nubes_en_formato_txt:            
                                        
                    puntos_nube,labels_semanticas_nube = leer_puntos_clases_Santarem_npy(self.root)


                puntos_nube = np.array(puntos_nube).astype(np.float64)     
                labels_semanticas_nube = np.array(labels_semanticas_nube).astype(np.float64)

                # print()
                # print()
                # print(puntos_nube.shape)
                # print(labels_semanticas_nube.shape)

                # import pdb
                # pdb.set_trace()


                Lista_puntos,Lista_labels_semanticas = [],[]
            
                # Vamos a desordenar ambos arrays del mismo modo:
                puntos_nube,labels_semanticas_nube = desordenar_dos_arrays(puntos_nube,labels_semanticas_nube)
            
                Lista_puntos.append(puntos_nube)
                Lista_labels_semanticas.append(labels_semanticas_nube)
            
                self.scene_points_list = Lista_puntos
                self.semantic_labels_list = Lista_labels_semanticas
                self.labelweights = np.ones(NUM_CLASSES)
            
            
            
            
            
            
            
            
            
            # En caso de estar haciendo predicciones sobre nubes sin etiquetar:
            else:
                # Impongo que solo para predicciones esto coja la escena completa:
                global ancho_de_celda
                
                os.chdir(root)
                
                Lista_puntos = []
                Lista_labels_semanticas = []
                
                
                self.labelweights = np.ones(NUM_CLASSES)
                # Lista_labels_semanticas.append(np.ones(NUM_CLASSES))
                
                # self.labelweights = Lista_labels_semanticas
                
                
                # Hay que leer la nube:
                if archivo.endswith('.npy'):
                    
                    os.chdir('Nube_artificial_9')
                    os.chdir('numpy_arrays')

                    
                    # Si la nube esta en formato .las o .laz hay que pasarla a .npy
                    # de la siguiente manera (PYTHON 3!!!)

                    '''
                    point_cloud=lp.file.File(archivo, mode="r")    
                    Lista_de_puntos = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
                    np.save('NUBE_ENTERA_PREDICCION.npy',Lista_de_puntos)
                    '''

                    point_cloud=np.load('Nube_artificial_9.npy')
                    if downsampleamos:
                        # Hacemos un downsampleo previo:
                        indices_seleccionados = np.random.choice(range(len(point_cloud)), 200000 , replace=False)
                        point_cloud = point_cloud[indices_seleccionados]


                    Lista_puntos.append(point_cloud)
                    Lista_labels_semanticas.append(np.full(shape=(len(point_cloud),),fill_value=1))

                    self.semantic_labels_list = Lista_labels_semanticas
                    self.scene_points_list = Lista_puntos


                    # print(Lista_de_puntos)
                    # print(Lista_labels_semanticas)

            
            
    def __getitem__(self, index):
        
        global ancho_de_celda
        
        # print('hola de nuevo')
        
        # print(self.scene_points_list)
        
        if self.root == '/home/lino/Documentos/programas_pruebas_varias/segmentacion_python/segmentacion_bosques/aumentacion_de_datos/Nubes_artificiales_generadas/nubes_buenas/dataset_carretera_nacional/PREDICT':

            point_set = self.scene_points_list[index][0]
            semantic_seg = self.semantic_labels_list[index].astype(np.int32)[0]
        
        else:
            
            point_set = self.scene_points_list[index]
            semantic_seg = self.semantic_labels_list[index].astype(np.int32)
            

        # IMPORTANTE!!!!!!
        # VAMOS A TRASLADAR LAS NUBES AL ORIGEN Y A ENCOGERLAS, YA
        # QUE LA RED LUEGO EMPLEA RADIOS MUY PEQUENHOS EN LAS ULTI-
        # MAS CAPAS (Y PUEDE SER QUE NO SE PILLEN SUFICIENTES PUN-
        # TOS AL NO ESTAR ENCOGIDA) Y LO DE TRASLADAR AL ORIGEN ES
        # PARA QUE LAS FUNCIONES SEAN SUAVIZADAS:
        
        
        # if self.split != "prediccion":
        #     print('holaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n')
        #     print(np.amin(point_set,axis=0))
        #     print('holaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n')
        
        
        
        
        
        if self.split == 'prediccion':
        
            if self.root == '/home/lino/Documentos/programas_pruebas_varias/segmentacion_python/segmentacion_bosques/aumentacion_de_datos/Nubes_artificiales_generadas/nubes_buenas/dataset_carretera_nacional/PREDICT':
            #     import pdb
            #     pdb.set_trace()
            
                # Encojo y traslado la nube pero no hago downsampling:
            
                factor_encoger = 790*5
                
                point_set = point_set/(factor_encoger)
                ####point_set = point_set-np.amin(point_set,axis=1)
                point_set = point_set-np.amin(point_set,axis=0)
                ancho_de_celda = 40

            else:
                
                
                factor_encoger = 5    
    
                
                point_set = point_set/(factor_encoger**2)
                point_set = point_set-np.amin(point_set,axis=0)
                ancho_de_celda = 40
        
        else:
            
            # print(np.max(point_set))
            # print(np.min(point_set))
            # print('testtttttttttttttttttttttttttttttttt')
            
            factor_encoger = 5    

            
            point_set = point_set/(factor_encoger**2)
            point_set = point_set-np.amin(point_set,axis=0)
        
        
            # print(np.max(point_set))
            # print(np.min(point_set))
            # print('testtttttttttttttttttttttttttttttttt')
        
        # Voy a introducir mi forma custom de pillar la celda.
        
        '''
        LOS DESARROLLADORES LO TENIAN ASI:
        coordmax = np.max(point_set,axis=0)
        coordmin = np.min(point_set,axis=0)
        smpmin = np.maximum(coordmax-[ancho_de_celda/2.,ancho_de_celda/2.,ancho_de_celda], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[ancho_de_celda/2.,ancho_de_celda/2.,ancho_de_celda])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            curmin = curcenter-[ancho_de_celda/4.,ancho_de_celda/4.,ancho_de_celda/2.]
            curmax = curcenter+[ancho_de_celda/4.,ancho_de_celda/4.,ancho_de_celda/2.]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3
            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg)==0:
                continue
            mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
            vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
            if isvalid:
                break
            
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        # choice son los indices que se escogen, por lo que guardamos ese vector
        # ACTUALIZACION: np.random.choice repite elementos si replace=True!!!
        indices_puntos_seleccionados = np.copy(choice)
        point_set = cur_point_set[choice,:]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        return point_set, semantic_seg, sample_weight, indices_puntos_seleccionados
    
        '''
        
        # print('paraparapara')
        
        # print(point_set)
        # print(semantic_seg)
        
        # if self.split == 'prediccion':
        #     import pdb
        #     pdb.set_trace()
        
        
        
        
        # if self.split == 'prediccion':
        #     print('INICIO')
        #     print(point_set.shape)
        #     print(semantic_seg.shape)
        #     print()
        #     print(point_set.take(2,1).max())
        #     print(point_set.take(2,1).min())
        #     with open("sample_prediccion_antes.npy", 'wb') as f:    
        #         np.save(f, point_set)
        #         np.save(f,semantic_seg)
        #     print('--------------------------------------------')
        
        # else:
        #     print('INICIO')
        #     print(point_set.shape)
        #     print(semantic_seg.shape)
        #     print()
        #     print(point_set.take(2,1).max())
        #     print(point_set.take(2,1).min())
        #     print('--------------------------------------------')
        
        
        coordmax = np.max(point_set,axis=0) # Regresa el maximo de cada columna en la lista de puntos
        coordmin = np.min(point_set,axis=0) # Regresa el minimo de cada columna en la lista de puntos

            
        smpmin = np.maximum(coordmax-[ancho_de_celda/2.,ancho_de_celda/2.,ancho_de_celda], coordmin)
        
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[ancho_de_celda/2.,ancho_de_celda/2.,ancho_de_celda])
        
        # print(smpsz)
        # print(smpsz)
        # print(smpsz)
        # print(smpsz)
        # print(smpsz)
        # print(smpsz)
        # print(smpsz)
        # print(smpsz)
        
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(100): # Iteramos 10 veces hasta conseguir un candidato valido:
            
            
            # if self.split == 'prediccion':
            #     import pdb
            #     pdb.set_trace()
            # else:
            #     print(coordmin,coordmax)
            #     print('NO PREDICCION OJOOOOOOOOOOOOOOOOOOO')
            
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            
            # print(curcenter)
                
            curmin = curcenter-[ancho_de_celda/4.,ancho_de_celda/4.,ancho_de_celda/2.]
            curmax = curcenter+[ancho_de_celda/4.,ancho_de_celda/4.,ancho_de_celda/2.]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set>=(curmin-ancho_de_celda/15.))*(point_set<=(curmax+ancho_de_celda/15.)),axis=1) == 3
            
            
            # print('HOLAAAAAAAAAAAAAAAAAAAAAAAAA')
            # print(curchoice)
            # print(curchoice.shape)
            
            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg)==0:
                continue
            mask = np.sum((cur_point_set>=(curmin-ancho_de_celda/300))*(cur_point_set<=(curmax+ancho_de_celda/300)),axis=1) == 3
            vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[10*ancho_de_celda,10*ancho_de_celda,20*ancho_de_celda])
            vidx = np.unique(vidx[:,0]*10*ancho_de_celda*20*ancho_de_celda+vidx[:,1]*(20*ancho_de_celda)+vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>-1)/len(cur_semantic_seg)>=0.7 
            if isvalid:
                break
        # print(isvalid)
        # print(cur_semantic_seg)
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        # choice son los indices que se escogen, por lo que guardamos ese vector
        # ACTUALIZACION: np.random.choice repite elementos si replace=True!!!
        indices_puntos_seleccionados = np.copy(choice)
        point_set = cur_point_set[choice,:]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        
        
        # if self.split == 'prediccion':
        #     print('FINAL')
        #     print(point_set.shape)
        #     print(semantic_seg.shape)
        #     print()
        #     print(point_set.take(2,1).max())
        #     print(point_set.take(2,1).min())
        #     with open("sample_prediccion_despues.npy", 'wb') as f:    
        #         np.save(f, point_set)
        #         np.save(f,semantic_seg)
        #     print('--------------------------------------------')
            
            
            
        #     import pdb
        #     pdb.set_trace()
        # else:
        #     print('FINAL')
        #     print(point_set.shape)
        #     print(semantic_seg.shape)
        #     print()
        #     print(point_set.take(2,1).max())
        #     print(point_set.take(2,1).min())
        #     print('--------------------------------------------')
        
        return point_set, semantic_seg, sample_weight, indices_puntos_seleccionados
    
    
    def __len__(self):
        return len(self.scene_points_list)


'''
class ScannetDatasetWholeScene():
    def __init__(self, root, npoints=8192, split='train',LINO=True,NUM_CLASSES=2,downsampleamos=True):
        self.NUM_CLASSES = NUM_CLASSES
        self.npoints = npoints
        self.root = root
        self.split = split
        
        if not LINO:
            self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
            with open(self.data_filename,'rb') as fp:
                self.scene_points_list = pickle.load(fp)
                self.semantic_labels_list = pickle.load(fp)
        else:
            
            if split != 'prediccion':
            
                Lista_puntos = []
                Lista_labels_semanticas = []
                
                # Voy a hacer que recorra todas las subcarpetas que haya en un di-
                # rectorio TRAIN (O TEST) y asi recopile todas las nubes en una
                # unica lista:
                
                os.chdir(root)
                
                # Estamos en la carpeta TRAIN (TEST), recorremos las subcarpetas:
                
                for archivo in os.listdir(os.getcwd()):
                    
                    if archivo.startswith('Nube_artificial'):
                        
                        os.chdir(archivo)
                        
                        # Por ahora voy a hacerlo para las nubes que solo tienen
                        # arboles y DTM:
                            
                        if not downsampleamos:
                            # Leemos sin hacer un downsampleo previo:
                            for subarchivo in os.listdir(os.getcwd()):                        
                                if subarchivo.endswith('arboles.npy'):
                                    puntos_arboles = np.load(subarchivo)
                                if subarchivo.endswith('DTM.npy'):
                                    puntos_DTM = np.load(subarchivo)
                                    
                        if downsampleamos:
                            # Hacemos un downsampleo previo:
                            for subarchivo in os.listdir(os.getcwd()):                        
                                if subarchivo.endswith('arboles.npy'):
                                    puntos_arboles = np.load(subarchivo)
                                    indices_seleccionados = np.random.choice(range(len(puntos_arboles)), 100000 , replace=False)
                                    puntos_arboles = puntos_arboles[indices_seleccionados]
                                if subarchivo.endswith('DTM.npy'):
                                    puntos_DTM = np.load(subarchivo)
                                    indices_seleccionados = np.random.choice(range(len(puntos_DTM)), 10000 , replace=False)
                                    puntos_DTM = puntos_DTM[indices_seleccionados]
                            
                        
                        # Ya estan leidos todos los arboles y todo el DTM. Voy a 
                        # crear unos arrays de etiquetas:
                            
                        etiquetas_arboles = np.full(shape=(1,len(puntos_arboles)),fill_value=0)
                        etiquetas_DTM = np.full(shape=(1,len(puntos_DTM)),fill_value=1)
                        
                        # Creamos ahora el array de todos los puntos de la nube y 
                        # el array de todas las labels semanticas de la nube:
                            
                        puntos_nube = np.vstack((puntos_arboles,puntos_DTM))
                        labels_semanticas_nube = np.vstack((etiquetas_arboles.T,etiquetas_DTM.T)).T[0]
                        
                        # Vamos a desordenar ambos arrays del mismo modo:
                        puntos_nube,labels_semanticas_nube = desordenar_dos_arrays(puntos_nube,labels_semanticas_nube)
                        
                        # Anhadimos los puntos y sus labels al dataset que estamos
                        # montando:
                            
                        Lista_puntos.append(puntos_nube)
                        Lista_labels_semanticas.append(labels_semanticas_nube)
                        
                        # Volvemos a la carpeta TRAIN (TEST)
                        os.chdir(root)
                        
                        
                        
                    
                # Adecuo las variables que usan los desarrolladores de PointNet a 
                # las mias:
                
                self.scene_points_list = Lista_puntos
                self.semantic_labels_list = Lista_labels_semanticas
            
            else:
                
                
                Lista_puntos = []
                Lista_labels_semanticas = []
                
                # Voy a hacer que recorra todas las subcarpetas que haya en un di-
                # rectorio TRAIN (O TEST) y asi recopile todas las nubes en una
                # unica lista:
                
                os.chdir(root)
                
                # Estamos en la carpeta TRAIN (TEST), recorremos las subcarpetas:
                
                
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                for subarchivo in os.listdir(os.getcwd()):                        
                    if subarchivo.endswith('.las'):
                        point_cloud=lp.file.File(subarchivo, mode="r")    
                        puntos_nube = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
                        try:
                            if len(point_cloud.red) != 0:
                                colores_aux = np.vstack((point_cloud.red,point_cloud.green,point_cloud.blue)).T
                                
                                maximo_R = np.max(point_cloud.red)
                                maximo_G = np.max(point_cloud.green)
                                maximo_B = np.max(point_cloud.blue)
                                
                                colores = np.vstack((colores_aux[:,0]/maximo_R,colores_aux[:,1]/maximo_G,colores_aux[:,2]/maximo_B)).T
                                
                                pcd.colors = o3d.utility.Vector3dVector(colores)
                                
                        except:
                            pass
                    if subarchivo.endswith('.laz'):
                        point_cloud=lp.file.File(subarchivo, mode="r")    
                        puntos_nube = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
                        try:
                            if len(point_cloud.red) != 0:
                                colores_aux = np.vstack((point_cloud.red,point_cloud.green,point_cloud.blue)).T
                                
                                maximo_R = np.max(point_cloud.red)
                                maximo_G = np.max(point_cloud.green)
                                maximo_B = np.max(point_cloud.blue)
                                
                                colores = np.vstack((colores_aux[:,0]/maximo_R,colores_aux[:,1]/maximo_G,colores_aux[:,2]/maximo_B)).T
                                
                                pcd.colors = o3d.utility.Vector3dVector(colores)
                                
                        except:
                            pass
                
                
                # Anhadimos los puntos y sus labels al dataset que estamos
                # montando:
                    
                Lista_puntos.append(puntos_nube)
                
                labels_semanticas_nube = np.zeros(len(puntos_nube))
                Lista_labels_semanticas.append(labels_semanticas_nube)
                
                # Volvemos a la carpeta TRAIN (TEST)
                os.chdir(root)
                        
                        
                        
                    
                # Adecuo las variables que usan los desarrolladores de PointNet a 
                # las mias:
                
                self.scene_points_list = Lista_puntos
                self.semantic_labels_list = Lista_labels_semanticas
                
                
            
        if split=='train':
    	    # labelweights = np.zeros(21)
    	    labelweights = np.zeros(NUM_CLASSES)
    	    for seg in self.semantic_labels_list:
        		# tmp,_ = np.histogram(seg,range(22))
        		tmp,_ = np.histogram(seg,range(NUM_CLASSES+1))
        		labelweights += tmp
    	    labelweights = labelweights.astype(np.float32)
    	    labelweights = labelweights/np.sum(labelweights)
    	    self.labelweights = 1/np.log(1.2+labelweights)
        if split=='test':
    	    # self.labelweights = np.ones(21)
    	    self.labelweights = np.ones(NUM_CLASSES)
        elif split=='prediccion':
    	    # self.labelweights = np.ones(21)
    	    self.labelweights = np.ones(NUM_CLASSES)
    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini,axis=0)
        coordmin = np.min(point_set_ini,axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        isvalid = False
        LISTA_INDICES = []
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*1.5,j*1.5,0]
                curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3
                cur_point_set = point_set_ini[curchoice,:]
                  
                  
                cur_semantic_seg = semantic_seg_ini[curchoice]
                  
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set>=(curmin-0.001))*(cur_point_set<=(curmax+0.001)),axis=1)==3
                
                # choice es el vector con los indices de los puntos seleccionados de la nube:
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                
                indices_puntos_seleccionados = np.copy(choice)
                LISTA_INDICES.append(indices_puntos_seleccionados)
                # print('ola k ase')
                # print(indices_puntos_seleccionados)
                
                point_set = cur_point_set[choice,:] # Nx3
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                if sum(mask)/float(len(mask))<0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N
                point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        if self.split != 'prediccion':
            return point_sets, semantic_segs, sample_weights
        else:
            LISTA_INDICES = np.array(LISTA_INDICES)
            return point_sets, semantic_segs, sample_weights, LISTA_INDICES
    def __len__(self):
        return len(self.scene_points_list)

'''

'''
class ScannetDatasetVirtualScan():
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        
        
        if not LINO:
            self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
            with open(self.data_filename,'rb') as fp:
                self.scene_points_list = pickle.load(fp)
                self.semantic_labels_list = pickle.load(fp)
        else:
            
            Lista_puntos = []
            Lista_labels_semanticas = []
            
            # Voy a hacer que recorra todas las subcarpetas que haya en un di-
            # rectorio TRAIN (O TEST) y asi recopile todas las nubes en una
            # unica lista:
            
            os.chdir(root)
            
            # Estamos en la carpeta TRAIN (TEST), recorremos las subcarpetas:
            
            for archivo in os.listdir(os.getcwd()):
                
                if archivo.startswith('Nube_artificial'):
                    
                    os.chdir(archivo)
                    
                    # Por ahora voy a hacerlo para las nubes que solo tienen
                    # arboles y DTM:
                        
                    for subarchivo in os.listdir(os.getcwd()):                        
                        if subarchivo.endswith('arboles.npy'):
                            puntos_arboles = np.load(subarchivo)
                        if subarchivo.endswith('DTM.npy'):
                            puntos_DTM = np.load(subarchivo)
                    
                    
                    # Ya estan leidos todos los arboles y todo el DTM. Voy a 
                    # crear unos arrays de etiquetas:
                        
                    etiquetas_arboles = np.full(shape=(1,len(puntos_arboles)),fill_value=1)
                    etiquetas_DTM = np.full(shape=(1,len(puntos_DTM)),fill_value=2)
                    
                    # Creamos ahora el array de todos los puntos de la nube y 
                    # el array de todas las labels semanticas de la nube:
                        
                    puntos_nube = np.vstack((puntos_arboles,puntos_DTM))
                    labels_semanticas_nube = np.vstack((etiquetas_arboles.T,etiquetas_DTM.T)).T[0]
                    
                    # Vamos a desordenar ambos arrays del mismo modo:
                    puntos_nube,labels_semanticas_nube = desordenar_dos_arrays(puntos_nube,labels_semanticas_nube)
                    
                    # Anhadimos los puntos y sus labels al dataset que estamos
                    # montando:
                        
                    Lista_puntos.append(puntos_nube)
                    Lista_labels_semanticas.append(labels_semanticas_nube)
                    
                    # Volvemos a la carpeta TRAIN (TEST)
                    os.chdir(root)
                    
                    
                    
                
            # Adecuo las variables que usan los desarrolladores de PointNet a 
            # las mias:
            
            self.scene_points_list = Lista_puntos
            self.semantic_labels_list = Lista_labels_semanticas
            
            
            
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
        elif split=='prediccion':
            self.labelweights = np.ones(21)
    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        sample_weight_ini = self.labelweights[semantic_seg_ini]
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in xrange(8):
            smpidx = scene_util.virtual_scan(point_set_ini,mode=i)
            if len(smpidx)<300:
                continue
                point_set = point_set_ini[smpidx,:]
            semantic_seg = semantic_seg_ini[smpidx]
            sample_weight = sample_weight_ini[smpidx]
            choice = np.random.choice(len(semantic_seg), self.npoints, replace=True)
            point_set = point_set[choice,:] # Nx3
            semantic_seg = semantic_seg[choice] # N
            sample_weight = sample_weight[choice] # N
            point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
            semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
            sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        return point_sets, semantic_segs, sample_weights
    def __len__(self):
        return len(self.scene_points_list)
'''



# Si diese problemas en 'train.py' descomentar lo de aqui abajo!!!!!!!!!!!:

'''
if __name__=='__main__':
    d = ScannetDatasetWholeScene(root = './data', split='test', npoints=8192)
    labelweights_vox = np.zeros(21)
    for ii in xrange(len(d)):
        print(ii)
        ps,seg,smpw = d[ii]
        for b in xrange(ps.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(ps[b,smpw[b,:]>0,:], seg[b,smpw[b,:]>0], res=0.02)
        tmp,_ = np.histogram(uvlabel,range(22))
        labelweights_vox += tmp
    print(labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32)))
    exit()
'''

