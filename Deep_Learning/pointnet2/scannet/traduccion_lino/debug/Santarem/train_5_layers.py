#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:46:08 2022

@author: lino
"""


#################################################################

#                   POINTNET++ PARA DUMMIES

#################################################################


# Instrucciones:
# 1) Crear un contenedor de docker o udocker con las especificaciones de siste-
#    ma del repositorio de GitHub de PointNet++ en la imagen.
# 2) Hacer modificaciones parametricas en este script en el apartado de PARAME-
#    TROS.
# 3) Correr el codigo en el contenedor creado.

import argparse
import math
from datetime import datetime
#import h5pyprovider
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys



#???
#------------------------------------------------------------------------------
# PARAMETROS:

# Le pido primero al usuario que de un nombre para guardar los datos de esta
# sesion:
nombre_sesion = 'WORKSHOP_VIERNES'
# hay que probarlo luego con la arquitectura por defecto (4 capas), 6 capas y,
# si vamos guay de tiempo, hacer lo mismo pero para otros valores de radios.

# Numero total de epocas sobre las cuales entrenar el modelo:
NUMERO_DE_EPOCAS = 100

# Va a haber dos downsampleos, uno que es el que se hace en el propio codigo de
# Pointnet++ (NUMERO_PUNTOS_VOXEL_PN) y otro que hago yo antes de meter las nu-
# bes en la red (NUMERO_PUNTOS_VOXEL_LINO):
NUMERO_PUNTOS_VOXEL_PN = 8192
NUMERO_PUNTOS_VOXEL_LINO = 65536
# NUMERO_PUNTOS_VOXEL_LINO = 200000

# Numero de nubes en las que se centrara el entrenamiento por cada vez (batches
# mas pequenhos hara que tarde y batches mas grandes requeriran mas memoria):
TAMANHO_BATCH = 1

# Modelo que vamos a emplear (ya que estamos con segmentacion pues usaremos el
# de segmentacion):
# MODELO = 'pointnet2_sem_seg_LINO_6_capas'
MODELO = 'pointnet2_sem_seg_LINO_5_capas'


# Por algun motivo especial interesa poner el nombre del usuario del PC (sale 
# en el log file pero bueno...)
HOSTNAME = socket.gethostname()


#------------------------------------------------------------------------------
#???

# ARREGLAMOS ALGUNOS PATHS IMPORTANTES (y terminamos de importar modulos):
'''
CESGA:
'''
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = '/mnt/netapp2/Store_uni/home/uvi/ir/lcc/scripts/deep_learning/Pointnet2/pointnet2'
# sys.path.append(BASE_DIR) # model
# sys.path.append(ROOT_DIR) # provider
# sys.path.append('/mnt/netapp2/Store_uni/home/uvi/ir/lcc/scripts/deep_learning/Pointnet2/pointnet2/models')
# sys.path.append('/mnt/netapp2/Store_uni/home/uvi/ir/lcc/scripts/deep_learning/Pointnet2/pointnet2/utils')

'''
LOCAL:
'''
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/home/lino/Documentos/PCD_DL_WORKSHOP/PCD_DL_WORKSHOP/Deep_Learning/pointnet2'
ROOT_DIR = '/home/lino/Documentos/PCD_DL_WORKSHOP/PCD_DL_WORKSHOP/Deep_Learning/pointnet2'
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append('/home/lino/Documentos/PCD_DL_WORKSHOP/PCD_DL_WORKSHOP/Deep_Learning/pointnet2/models')
sys.path.append('/home/lino/Documentos/PCD_DL_WORKSHOP/PCD_DL_WORKSHOP/Deep_Learning/pointnet2/utils')



import provider
import tf_util
import pc_util
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import scannet_dataset

# OPCIONES CUSTOMIZABLES A LA HORA DE LANZAR EL ENTRENAMIENTO EN CONSOLA (Y SUS
# VALORES PREDETERMINADOS, LOS QUE PUSIMOS EN EL APTDO DE PARAMETROS, VAYA):
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default=MODELO, help='Model name [default: pointnet2_sem_seg]')
parser.add_argument('--log_dir', default='log_%s'%nombre_sesion, help='Log dir [default: log_%s]'%nombre_sesion)
parser.add_argument('--num_point', type=int, default=NUMERO_PUNTOS_VOXEL_LINO, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=NUMERO_DE_EPOCAS, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=TAMANHO_BATCH, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()






# Contador de EPOCA en la que nos encontramos:
EPOCH_CNT = 0

# Cogemos los valores preestablecidos anteriormente como variables:
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)


# Creo una variable del path absoluto del LOG_DIR:
ruta_actualisima = os.getcwd()
os.chdir(LOG_DIR)
RUTA_LOG_ABSOLUTA = os.getcwd()
os.chdir(ruta_actualisima)


# Se intentaran copiar unos archivos a modo backup, si falla que no cunda el
# panico que no es muy importante:
os.system('cp %s %s' % (MODEL_FILE, RUTA_LOG_ABSOLUTA)) # bkp of model def
os.system('cp train.py %s' % (RUTA_LOG_ABSOLUTA)) # bkp of train procedure

# Abrimos en el directorio 'log' el archivo log (xD), que es donde van a estar 
# los mensajes que se expulsan durante el entrenamiento:
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# Por ahora esto de BN no se que es:
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99




# Este parametro hace referencia a si uso mis propios datos o los del paper de
# los desarrolladores de Pointnet++:
LINO = True

# En funcion de los datos que empleemos especificamos las rutas a los mismos:
if not LINO:
    NUM_CLASSES = 2
    # Shapenet official train/test split
    DATA_PATH = os.path.join(ROOT_DIR,'data','scannet_data_pointnet2')
    TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train',LINO=LINO,NUM_CLASSES=NUM_CLASSES)
    TEST_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test',LINO=LINO,NUM_CLASSES=NUM_CLASSES)
    TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test',LINO=LINO,NUM_CLASSES=NUM_CLASSES)
else:
    # Usamos mis propios datos:
    # DATASET_PATH = '/home/lino/Documentos/programas_pruebas_varias/segmentacion_python/segmentacion_bosques/aumentacion_de_datos/Nubes_artificiales_generadas/nubes_buenas/dataset_carretera_nacional'
    DATASET_PATH = '/home/lino/Documentos/PCD_DL_WORKSHOP/PCD_DL_WORKSHOP/data/synthetic_point_clouds/SYNTHETIC_DATASET'
    
    TRAIN_DATA_PATH = DATASET_PATH +'/TRAIN'
    TEST_DATA_PATH = DATASET_PATH +'/TEST'    
    # PREDICT_DATA_PATH = '/home/lino/Documentos/programas_pruebas_varias/segmentacion_python/segmentacion_bosques/aumentacion_de_datos/Nubes_artificiales_generadas/nubes_buenas/dataset_carretera_nacional/PREDICT'
    PREDICT_DATA_PATH = DATASET_PATH + '/PREDICT'
    
    NUMERO_NUBES_ENTRENAMIENTO = len(os.listdir(TRAIN_DATA_PATH))
    NUMERO_NUBES_TESTEO = len(os.listdir(TEST_DATA_PATH))
    
    # NUMERO_NUBES_ENTRENAMIENTO = len(os.listdir(TRAIN_DATA_PATH))-45 # nacionalllllllllllllllll
    # NUMERO_NUBES_TESTEO = len(os.listdir(TEST_DATA_PATH))-10
    
    
    
    NUMERO_NUBES_PREDICCION = 1
    # Para saber el numero de clases con las que se va a hacer el entrenamiento,
    # abrimos el fichero 'informacion_dataset.txt' generado en el directorio de las
    # nubes sinteticas (un nivel encima de las carpetas TRAIN y TEST ojo!!!)
    os.chdir(DATASET_PATH)
    
    tipo_de_nubes = 'bosque'
    NUM_CLASSES = 2
    
    # Voy a pillar el numero en el que empieza cada nube en cada dataset:
    os.chdir(TRAIN_DATA_PATH)
    
    Lista_auxiliar = []
    for archivo in os.listdir(os.getcwd()):
        if archivo.startswith('Nube_artificial'):
            Lista_auxiliar.append(int(archivo[16:]))
    contador_dataset_entrenamiento = np.min(Lista_auxiliar)
    contador_dataset_entrenamiento_original = int(np.copy(contador_dataset_entrenamiento))
    
    os.chdir(TEST_DATA_PATH)
    
    Lista_auxiliar = []
    for archivo in os.listdir(os.getcwd()):
        if archivo.startswith('Nube_artificial'):
            Lista_auxiliar.append(int(archivo[16:]))
    contador_dataset_testeo = np.min(Lista_auxiliar)
    contador_dataset_testeo_original = int(np.copy(contador_dataset_testeo))


    os.chdir(ruta_actualisima)


# Las clases de Scannet_Dataset vienen explicadas en el archivo con el mismo
# nombre del directorio de este mismo script.








# Definimos la funcion que se encarga de escribir en el log file:
def log_string(out_str):
    # Anhadimos el string que queremos meter y una linea nueva en blanco:
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Definimos la funcion que se encarga de sustraer el learning rate en un momen-
# to especifico del entrenamiento:
def get_learning_rate(batch):
    # El como funciona aun no lo tengo claro muy bien:
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

# Al igual que los parametros de antes aun no se muy bien que hace esto:
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


# Definimos la funcion del entrenamiento en si:
def train():
    with tf.Graph().as_default():
        # Escribimos el indice de la GPU seleccionada (en el caso de tener un 
        # equipo con mas de una grafica pues sera 0 u otro valor):
        with tf.device('/gpu:'+str(GPU_INDEX)):
            
            # Creamos los placeholders de las nubes de puntos, las etiquetas y
            # los pesos. Un placeholder es basicamente un espacio de memoria
            # que va a ser rellenado con un tensor:
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            
            # Creamos otro placeholder tipo booleano en el que alojaremos un
            # valor logico que nos ayudara a diferenciar entre cuando entrena-
            # mos y cuando no:
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Notas de los desarrolladores de PN++:
            '''
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            '''
            
            # Creamos dos variables de Tensorflow, una asociada a un batch ge-
            # nerico y otra al bn_decay.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            # Anhadimos el bn_decay como un escalar a la tabla summary de TF:
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Vamos a obtener el modelo y las perdidas.
            
            # Sustraemos dos arrays, uno con las predicciones y otro un poco
            # mas especial que explico justo despues:
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            # end_points es un diccionario con dos elementos, uno que es la nu-
            # de puntos que introducimos y otro con las features sustraidas, al
            # final del modelo (ver pointnet2_sem_seg.py).
            
            # Obtenemos las perdidas como forma de array:
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
            # Anhadimos las perdidas como un escalar a la tabla summary de TF:
            tf.summary.scalar('loss', loss)

            # Comparamos dos tensores, uno con las predicciones mas probables
            # (de ahi lo del maximo) y otro con las etiquetas reales:
            # (Ojo a las explicaciones de justo despues)
            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            # tf.argmax(array, indice_dimension) --->  Devuelve un array con 
            #                                          los indices de los ele-
            #                                          mentos maximos del se-
            #                                          gundo atributo 'indice_dimension'
            
            # tf.to_int64('tensor en placeholder') ---> Devuelve el mismo ten-
            #                                           sor pero como int64
            
            # Sacamos el valor de la precision en este batch:
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            # Anhadimos las precision como un escalar a la tabla summary de TF:
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Vamos a obtener el operador del entrenamiento
            
            # Cogemos el learning rate segun la funcion previamente explicada:
            learning_rate = get_learning_rate(batch)
            # Anhadimos el learning_rate como un escalar a la tabla summary de TF:
            tf.summary.scalar('learning_rate', learning_rate)
            
            # Especificamos el optimizador que vamos a usar de la coleccion de TF:
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Nota de los desarrolladores de PN++:
            '''
            # Add ops to save and restore all the variables.
            '''
            # Creamos un saver, que sirve para crear checkpoints durante el en-
            # trenamiento y asi poder entrenar/testear/etc el modelo desde una
            # epoca concreta:
            saver = tf.train.Saver()
        
        
        
        # Vamos a crear una sesion.
        config = tf.ConfigProto()
        # Generalmente se utiliza para configurar los parámetros de la sesión 
        # al crear una sesión.

        # A continuacion algunos parametros de configuracion de la sesion que
        # explico despues:
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # · config.gpu_options.allow_growth: Se usa para evitar que la GPU quede
        #   sin recursos de memoria.
        # · config.allow_soft_placement: Si no se usa TF expulsara un error.
        # · config.log_device_placement: Basicamente hace que salgan en consola
        #   unos mensajes que dicen donde esta mapeada cada operacion.
        
        # Creamos la sesion con toda la configuracion que acabamos de hacer:
        sess = tf.Session(config=config)
        
                
        # Fusionamos todos los 'summaries' recolectados hasta ahora:
        merged = tf.summary.merge_all()
        
        
        # Add summary writers

        # Anhadimos un Writer, que lo que hace es escribir en un archivo todos
        # los logs (acorde con el protocolo especificado en el summary). Anha-
        # mos un Writer para el entrenamiento y otro para el testeo:
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Iniciamos las variables y las vinculamos a la sesion creada:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Comentario de los desarrolladores de PN:
        '''
        #sess.run(init, {is_training_pl: True})
        '''
        
        # Creamos un diccionario con todas las operaciones (ops):
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        # A modo referencia empezamos poniendo una variable de mejor precision
        # negativa (y durante el entrenamiento se actualiza):
        best_acc = -1
        
        # Iteramos a lo largo de las epocas especificadas por el usuario:
        for epoch in range(MAX_EPOCH):
            
            
            log_string('     \n')
            # Mostramos la epoca actual en pantalla:
            log_string('**** EPOCH %03d ****' % (epoch))
            log_string('     \n')

            sys.stdout.flush()
            # sys.stdout.flush() fuerza a que aparezca en pantalla todos los
            # logs pendientes en el codigo hasta ese momento.

            # En cada epoca ejecutamos la funcion de entrenar durante una epoca
            # (train_one_epoch, definida y explicada mas adelante):
            train_one_epoch(sess, ops, train_writer,RUTA_LOG_ABSOLUTA,epoca=epoch)
            
            # Cada 5 epocas nos paramos y pedimos hacer una prediccion sobre da-
            # tos no vistos por la red para evaluar la precision actual:
            #if epoch%5==0:
            if epoch%1==0:   
                    acc = eval_one_epoch(sess, ops, test_writer,epoch,RUTA_LOG_ABSOLUTA)
                    # acc = eval_whole_scene_one_epoch(sess, ops, test_writer,epoch,RUTA_LOG_ABSOLUTA)
                    
                    

                    # Tambien me saco una prediccion sobre nube real:
                    eval_one_epoch_PREDICCION_NUBE_REAL(sess, ops, test_writer, epoch, RUTA_LOG_ABSOLUTA)

                    
                    
            # Si la precision adquirida (actualizada cada 5 epocas) es mejor que
            # la previa la actualizamos:
            if acc > best_acc and acc != 'error':
                best_acc = acc
                
                # Ademas, guardamos el modelo actual, ya que es el que por ahora
                # obtuvo una precision mayor (al final del entrenamiento es muy
                # posible encontrarnos con varios de estos modelos, es cuestion
                # de graficar precisiones y losses vs epocas y seleccionar el
                # que mas nos convenza):
                save_path = saver.save(sess, os.path.join(RUTA_LOG_ABSOLUTA, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)
            
            
            # Cada 10 epocas guardamos las variables (el modelo, vaya):
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(RUTA_LOG_ABSOLUTA, "%i_epoch_model.ckpt"%epoch))
                log_string("Model saved in file: %s" % save_path)
'''

###
# Forma por defecto de POINTNET++
###

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
    batch_smpw[i,:] = smpw

    dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
    batch_data[i,drop_idx,:] = batch_data[i,0,:]
    batch_label[i,drop_idx] = batch_label[i,0]
    batch_smpw[i,drop_idx] *= 0
    return batch_data, batch_label, batch_smpw

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
    batch_smpw[i,:] = smpw
    return batch_data, batch_label, batch_smpw

'''

# A continuacion se definen mis formas custom de leer batches. Lo que buscamos 
# en estas funciones es obtener el batch (con el numero de nubes especificado en 
# 'TAMANHO_BATCH') y toda suinformacion, es decir:
# · batch_data: Tensor donde cada elemento es el array de una nube de puntos.
# · batch_label: Array donde cada elemento es el indice de cada batch.
# · batch_smpw: (smpw==sample weight) Array con los pesos de cada batch.
# · indices_puntos_seleccionados

# Definicion de como sustraer el batch que queremos del dataset pero aplicandole
# un Dropout. Un Dropout lo que hace es igualar a 0 los pesos de alguna capa, es
# decir, omitir o "abandonar" neuronas DE FORMA ALEATORIA. 
def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    indices_puntos_seleccionados = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps,seg,smpw,indices = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
        indices_puntos_seleccionados[i,:] = indices
    
        dropout_ratio = np.random.random()*0.875 # [0-0.875]
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_smpw[i,drop_idx] *= 0
        indices_puntos_seleccionados[i,drop_idx] *= 0
    return batch_data, batch_label, batch_smpw, indices_puntos_seleccionados

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    indices_puntos_seleccionados = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        try:
            ps,seg,smpw, indices = dataset[idxs[i+start_idx]]
            indices_puntos_seleccionados[i,:] = indices
            batch_data[i,...] = ps
            batch_label[i,:] = seg
            batch_smpw[i,:] = smpw
            return batch_data, batch_label, batch_smpw, indices_puntos_seleccionados
        except:
            ps,seg,smpw = dataset[idxs[i+start_idx]]
            batch_data[i,...] = ps
            batch_label[i,:] = seg
            batch_smpw[i,:] = smpw
            return batch_data, batch_label, batch_smpw



# Aqui definimos el proceso de entrenamiento de una epoca:

def train_one_epoch(sess, ops, train_writer,RUTA_LOG_ABSOLUTA,epoca):
    
    """ 
    NOTA DE LOS DESARROLLADORES:
    ops: dict mapping from string to tf ops
    """
    
    for ll in range(NUMERO_NUBES_ENTRENAMIENTO):
    
        global contador_dataset_entrenamiento
        
        TRAIN_DATASET = scannet_dataset.ScannetDataset(root=TRAIN_DATA_PATH, npoints=NUM_POINT, split='train',LINO=LINO,NUM_CLASSES=NUM_CLASSES,downsampleamos=False,contador_dataset=contador_dataset_entrenamiento,tamanho_batch = BATCH_SIZE,tipo_de_nubes=tipo_de_nubes,algunas_clases_fusionadas=True)
        contador_dataset_entrenamiento += BATCH_SIZE
        
        
        is_training = True
    
        
        # Desordenamos las muestras de las nubes de forma aleatoria:
        train_idxs = np.arange(0, len(TRAIN_DATASET))
        np.random.shuffle(train_idxs)
        num_batches = len(TRAIN_DATASET)/BATCH_SIZE
        
        # Mostramos en pantalla la fecha del momento actual:
        log_string(str(datetime.now()))
    
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        
        # Iteramos para cada batch de nubes de puntos:
        for batch_idx in range(num_batches):
            
            # Indices del batch inicial y final:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Cargamos los datos de este batch:
            batch_data, batch_label, batch_smpw, indices_puntos_seleccionados = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
    
            # print(len(batch_data))
            # print(len(batch_data[0]))
    
    
    
            # global contador_javi
    
            # ruta_ruptura = os.getcwd()
            # os.chdir(RUTA_LOG_ABSOLUTA)
            # with open("datos_cargados_%i.npy"%contador_javi, 'wb') as f:    
            #     np.save(f, batch_data)
            #     np.save(f,batch_label)
            # contador_javi += 1
    
            # Hacemos data augmentation rotando las nubes de puntos de cada batch
            # mediante la funcion rotate_point_cloud_z del modulo provider:
            aug_data = provider.rotate_point_cloud_z(batch_data)
            # import pdb; pdb.set_trace()
            
            
            # Introduciremos en la red los datos en forma de diccionario con los 
            # datos aumentados de cada batch, los indices de cada batch, los pesos 
            # la variable switch de entrenamiento (booleana):        
            feed_dict = {ops['pointclouds_pl']: aug_data,
                         ops['labels_pl']: batch_label,
                         ops['smpws_pl']:batch_smpw,
                         ops['is_training_pl']: is_training,}
    
            # INTRODUCIMOS DATOS EN LA RED:
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            
            
            # Recordemos que es cada cosa:
            # · merged: Los summaries todos fusionados.
            # · step: El batch en el que nos encontramos.
            # · train_op: El optimizador que empleamos en el entrenamiento (Adam o
            #             Momentum, especificado arriba).
            # · loss: Las perdidas en esa fase del entrenamiento.
            # . pred: El array de predicciones en esa fase del entrenamiento.
            # · feed_dict: Los datos con los que alimentamos la red en forma de 
            #             diccionario.
            
            # Escribimos el summary de esta fase de entrenamiento:
            train_writer.add_summary(summary, step)
            
            # Cogemos como valores de prediccion aquellos que se repitan mas:
            pred_val = np.argmax(pred_val, 2)
            # Este pred_val tendra un numero de filas igual al numero de nubes en
            # el batch. Cada elemento de ese array estara constituido a su vez por
            # un array columna donde cada elemento se corresponde con la etiqueta 
            # de predicha para cada punto de la nube.
            
            
            # print(pred_val)
            
            
            
            # Vemos cuantos puntos tienen una label de prediccion igual a las label
            # verdaderas y almacenamos ese valor (ira incrementando en cada batch):
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            
            # print(total_correct)
            
            # Actualizamos el numero total de puntos vistos:
            total_seen += (BATCH_SIZE*NUM_POINT)
            
            # print(total_seen)
            
            # Actualizamos las perdidas totales:
            loss_sum += loss_val
            
            # Si el batch en el que nos encontramos es multiplo de 10 entonces mos-
            # tramos en pantalla la informacion de precision y perdidas en ese mo-
            # mento y reseteamos esas variables:
            # if (batch_idx+1)%10 == 0:
            # log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / BATCH_SIZE))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            
                # OJO QUE IGUAL TENGO QUE DESCOMENTAR AQUI ABAJO:
                # total_correct = 0
                # total_seen = 0
                # loss_sum = 0
                
                
            
                
            #######################################################################
            #                        PARON PARA VISUALIZAR
            # Voy a guardar aqui los arrays de prediccion y la propia nube de pun-
            # tos para dibujarla externamente y ver que pasa:
            
            ruta_ruptura = os.getcwd()
            
            os.chdir(RUTA_LOG_ABSOLUTA)
            
            
                
            if epoca == 0 and ll == 0:
                f = open("%s_accuracies_y_losses_ENTRENAMIENTO.txt"%nombre_sesion, "w")
                f.write("ACCURACIES    LOSSES    EPOCH    BATCH    NUBE_ID\n")
                f.write(str(total_correct / float(total_seen))+'    '+str(loss_sum / float(num_batches))+'    '+str(epoca)+'    '+str(batch_idx)+'    '+str(ll)+'\n')
                f.close()
              
            else:
                f = open("%s_accuracies_y_losses_ENTRENAMIENTO.txt"%nombre_sesion, "a")
                f.write(str(total_correct / float(total_seen))+'    '+str(loss_sum / float(num_batches))+'    '+str(epoca)+'    '+str(batch_idx)+'    '+str(ll)+'\n')
                f.close()
            
            
            
            with open("prediccion_entrenamiento.npy", 'wb') as f:    
                np.save(f, aug_data)
                np.save(f,pred_val)
                
            # # # Para cargar esos arrays haríamos así:
                
            # import numpy as np
            # import open3d as o3d
            # import os
            # os.chdir('/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_ancho_celda_5_repeticion')
            # with open("prediccion_entrenamiento.npy", 'rb') as f:
            #     aug_data = np.load(f)[0]
            #     pred_val = np.load(f)[0]
                    
            # indices_arboles = np.where(pred_val == 0)
            # indices_DTM = np.where(pred_val == 1)
            
            # colores = np.zeros(shape=aug_data.shape)
            # colores[indices_arboles] = [0,1,0]
            # colores[indices_DTM] = [1,0,0]
            
            # import open3d as o3d
            # nube_clasificada = o3d.geometry.PointCloud()
            # nube_clasificada.points = o3d.utility.Vector3dVector(aug_data)
            # nube_clasificada.colors = o3d.utility.Vector3dVector(colores)
            # o3d.visualization.draw(nube_clasificada)
            
                
                
                
                
                
            #######################################################################
            
            # RESULTADOS PAPER:
            
            TP = [0 for _ in range(NUM_CLASSES)]
            FP = [0 for _ in range(NUM_CLASSES)]
            FN = [0 for _ in range(NUM_CLASSES)]
            MA = [0 for _ in range(NUM_CLASSES)]
            IoU = [0 for _ in range(NUM_CLASSES)]
        
            for l in range(NUM_CLASSES):   
                
                # import pdb
                # pdb.set_trace()
                
                # TP:
                TP[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
                # FN:
                FN[l] +=  np.sum((pred_val!=l) & (batch_label==l) & (batch_smpw>0))
                # FP:
                FP[l] += np.sum((pred_val==l) & (batch_label!=l) & (batch_smpw>0))
    
                IoU[l] += TP[l]/float(TP[l]+FP[l]+FN[l])
                MA[l] += total_correct/float(total_seen)
        
            MA = np.nanmean(MA)
            MIoU = np.nanmean(IoU)
        
        
            # Overall accuracy (OA):
            OA = total_correct / float(total_seen)
            log_string('train point accuracy: %f'% OA)
            
            # Mean Accuracy (MA):
            log_string('MA: %f'% (MA))
            
            # Mean Loss:
            Mean_Loss = loss_sum / float(num_batches)
            log_string('Mean Loss: %f'% Mean_Loss)
            
            # MIoU:
            log_string('MIoU: %f'% (MIoU))
        
        
        
        
            # RESULTADOS PAPER:
            # ll == indice de la nube en el set de entrenamiento
            if epoca == 0 and ll == 0:
                f = open("RESULTADOS_PAPER_%s_TRAIN.txt"%nombre_sesion, "w")
                f.write("Overall Accuracy (OA)    Mean Accuracy (MA)    Mean Loss    MIoU    Epoch    NUBE_ID\n")
                f.write(str(OA)+'    '+str(MA)+'    '+str(Mean_Loss)+'    '+str(MIoU)+'    '+str(epoca)+'    '+str(ll)+'\n')
                f.close()
            else:
                f = open("RESULTADOS_PAPER_%s_TRAIN.txt"%nombre_sesion, "a")
                f.write(str(OA)+'    '+str(MA)+'    '+str(Mean_Loss)+'    '+str(MIoU)+'    '+str(epoca)+'    '+str(ll)+'\n')
                f.close()
                    
            
            # # Volvemos a donde estabamos:
            os.chdir(ruta_ruptura)
            
            #######################################################################
                
                
                
                
                
                
                
                
                
                
                
    # Reseteo para la siguiente epoca:
    contador_dataset_entrenamiento = contador_dataset_entrenamiento_original






# Definicion de como se evalua en nubes cortadas:
''' Comentario de los desarrolladores:   
# evaluate on randomly chopped scenes
'''
def eval_one_epoch(sess, ops, test_writer,epoch,RUTA_LOG_ABSOLUTA):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    
    
    for ll in range(NUMERO_NUBES_TESTEO):
    
        global contador_dataset_testeo
        
        TEST_DATASET = scannet_dataset.ScannetDataset(root=TEST_DATA_PATH, npoints=NUM_POINT, split='test',LINO=LINO,NUM_CLASSES=NUM_CLASSES,downsampleamos=False,contador_dataset=contador_dataset_testeo,tamanho_batch = BATCH_SIZE, tipo_de_nubes=tipo_de_nubes,algunas_clases_fusionadas=True)
        contador_dataset_testeo += BATCH_SIZE
    
    
    
    
    
        is_training = False
    
    
    # TEST_DATASET = scannet_dataset.ScannetDataset(root=TEST_DATA_PATH, npoints=NUM_POINT, split='test',LINO=LINO,NUM_CLASSES=NUM_CLASSES,downsampleamos=True, contador_dataset=contador_dataset_testeo, tamanho_batch=BATCH_SIZE)
        
        # TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=TEST_DATA_PATH, npoints=NUM_POINT, split='test',LINO=LINO,NUM_CLASSES=NUM_CLASSES,downsampleamos=True)
        
        
        
        # Indices de cada nube en el set de testeo:
        test_idxs = np.arange(0, len(TEST_DATASET))
        
        # Numero de batches que tendremos en el set de testeo:
        num_batches = len(TEST_DATASET)/BATCH_SIZE
    
    
        # log_string(str(contador_dataset_testeo))
        # log_string(str(BATCH_SIZE))
        # log_string('holaaaaaaaaaaaaaaaaaaaaaaaaaa')
    
    
    
        # Definimos estas variables que actualizaremos en un rato:
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        
        TP = [0 for _ in range(NUM_CLASSES)]
        FP = [0 for _ in range(NUM_CLASSES)]
        FN = [0 for _ in range(NUM_CLASSES)]
        MA = [0 for _ in range(NUM_CLASSES)]
        IoU = [0 for _ in range(NUM_CLASSES)]

    
        # Definimos estas variables tambien que haran referencia a las mismas can-
        # tidades pero medidas en cada "caja" (recordemos que PointNet++ hace esta 
        # evaluacion por bloques que recorta de la nube):
        total_correct_vox = 0
        total_seen_vox = 0
        total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
        total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]
        
        # Mostramos en pantalla la fecha del momento actual y que nos encontramos 
        # en la fase de evaluacion de la epoca actual:
        log_string(str(datetime.now()))
        log_string('---- EPOCH %03d EVALUATION ----'%(epoch))
    
        # Creamos unos arrays de ceros que haran referencia a los pesos de cada la-
        # bel:
        labelweights = np.zeros(NUM_CLASSES+1)
        labelweights_vox = np.zeros(NUM_CLASSES+1)
        
        # Iteramos para cada batch de nubes en el set de testeo:
        for batch_idx in range(num_batches):
           
    
            # Indices del batch inicial y final:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Cargamos los datos de este batch de testeo:
            batch_data, batch_label, batch_smpw, indices_puntos_seleccionados = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
    
            # Hacemos data augmentation rotando las nubes de puntos de cada batch
            # mediante la funcion rotate_point_cloud_z del modulo provider:
            aug_data = provider.rotate_point_cloud_z(batch_data)
    
            # Introduciremos en la red los datos en forma de diccionario con los 
            # datos aumentados de cada batch, los indices de cada batch, los pesos 
            # la variable switch de entrenamiento (booleana):        
            feed_dict = {ops['pointclouds_pl']: aug_data,
                         ops['labels_pl']: batch_label,
                         ops['smpws_pl']: batch_smpw,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            # Recordemos que es cada cosa:
            # · merged: Los summaries todos fusionados.
            # · step: El batch en el que nos encontramos.
            # · loss: Las perdidas en esa fase del entrenamiento.
            # . pred: El array de predicciones en esa fase del entrenamiento.
            # · feed_dict: Los datos con los que alimentamos la red en forma de 
            #             diccionario.
            
            # Escribimos el summary de esta fase de testeo:
            test_writer.add_summary(summary, step)
            
            # Cogemos como valores de prediccion aquellos que se repitan mas:
            pred_val = np.argmax(pred_val, 2) # BxN
            # print(np.count_nonzero(batch_label))
            



            # Voy a guardar aqui los arrays de prediccion y la propia nube de pun-
            # tos para dibujarla externamente y ver que pasa:
            
            ruta_ruptura = os.getcwd()
            
            os.chdir(RUTA_LOG_ABSOLUTA)
            
            with open("prediccion_nube_epoca_%i.npy"%epoch, 'wb') as f:    
                np.save(f, aug_data)
                np.save(f,pred_val)
                
                
            
            # #######################################################################
            # #                        PARON PARA VISUALIZAR
            # # Para cargar esos arrays haríamos así:
                
            # import numpy as np
            # import open3d as o3d
            # import os
            # os.chdir('/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_prueba_autopista_2')
            # with open("prediccion_entrenamiento.npy", 'rb') as f:
            #     aug_data = np.load(f)[0]
            #     pred_val = np.load(f)[0]
                    
            # indices_arboles = np.where(pred_val == 0)
            # indices_DTM = np.where(pred_val == 1)
            # indices_carretera = np.where(pred_val == 2)
            # indices_talud = np.where(pred_val == 3)
            # indices_senhal = np.where(pred_val == 4)
            # indices_barrera_quitamiedos_1 = np.where(pred_val == 5)
            # indices_barrera_quitamiedos_2 = np.where(pred_val == 6)
            # indices_arcen = np.where(pred_val == 7)
            # indices_mediana = np.where(pred_val == 8)
            # indices_barrera_jersey = np.where(pred_val == 9)
            
            # colores = np.zeros(shape=aug_data.shape)
            # colores[indices_arboles] = [0,1,0]
            # colores[indices_DTM] = [1,0,0]
            # colores[indices_carretera] = [0,0,0]
            # colores[indices_talud] = [1,0.5,0.5]
            # colores[indices_senhal] = [0,0,1]
            # colores[indices_barrera_quitamiedos_1] = [0.4,0.4,0.4]
            # colores[indices_barrera_quitamiedos_2] = [0.4,0.4,0.4]
            # colores[indices_arcen] = [1,1,0]
            # colores[indices_mediana] = [0.1,0.21,0.1]
            # colores[indices_barrera_jersey] = [0.7,0.4,0.1]
            
            # import open3d as o3d
            # nube_clasificada = o3d.geometry.PointCloud()
            # nube_clasificada.points = o3d.utility.Vector3dVector(aug_data)
            # nube_clasificada.colors = o3d.utility.Vector3dVector(colores)
            # o3d.visualization.draw(nube_clasificada)
            
            
            
            #######################################################################
            # Volvemos a donde estabamos:
            os.chdir(ruta_ruptura)
            
    
            
            # Vemos cuantos puntos tienen una label de prediccion igual a las label
            # verdaderas y almacenamos ese valor (ira incrementando en cada batch):
            correct = np.sum((pred_val == batch_label) & (batch_smpw>0))
            '''
            LOS DESARROLLADORES LO TENIAN ASI:
            correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0))
            '''
            
            total_correct += correct
            
            # Actualizamos el numero total de puntos vistos:
            total_seen += np.sum((batch_label>-1) & (batch_smpw!=0))
            '''
            LOS DESARROLLADORES LO TENIAN ASI:
            total_seen += np.sum((batch_label>0) & (batch_smpw>0))
            '''
            # Actualizamos las perdidas totales:
            loss_sum += loss_val
            
            # Hasta aqui todo mas o menos igual a como haciamos en la fase de en-
            # trenamiento. Ahora vamos a crear un histograma con un total de bins 
            # igual a la variable NUMERO_BINS y en el eje Y el factor de repeticion
            # de cada label:
            NUMERO_BINS = NUM_CLASSES + 2
            tmp,_ = np.histogram(batch_label,range(NUMERO_BINS))
            # Aclaracion:
            # · tmp: Array donde cada elemento es el numero de veces que se repite
            #        el valor de cada bin.
            # · _: Array con los valores de cada bin (0,1,2,3,4,5, ...).
            
            # Actualizamos los pesos de cada etiqueta:
            labelweights += tmp
            
            
            os.chdir(RUTA_LOG_ABSOLUTA)
            
            # Voy a guardar los pesos para pintarlos externamente:
            with open("label_weights_epoca_%i.npy"%epoch, 'wb') as f:    
                np.save(f, labelweights)
                
            #######################################################################
            #                        PARON PARA VISUALIZAR
            # Una vez guardados los pesos, voy a leerlos externamente para ver que
            # pasa:
            
            # import numpy as np
            
            # os.chdir('/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_prueba_300_epocas')
            
            # Lista_labelweights = []
            # for contador_epoca in range(len(os.listdir(os.getcwd()))):
            #     try:
            #         with open('label_weights_epoca_%i.npy'%contador_epoca, 'rb') as f:
            #             labelweights = np.load(f)
            #             print(labelweights)
            #         Lista_labelweights.append(labelweights)
            #     except FileNotFoundError:
            #         pass
            
            #######################################################################
    
    
    
    
            os.chdir(ruta_ruptura)
            
            # Iteramos sobre el numero de clases para ver las precisiones de seg-
            # mentacion por clases:
            for l in range(NUM_CLASSES):
                
                
                # import pdb
                # pdb.set_trace()
                
                # Actualizamos el numero total de clases vistas:
                total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
                
                
                # Actualizamos el numero total de clases correctamente segmentadas:
                # TP:
                total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
                TP += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
                # FN:
                FN[l] +=  np.sum((pred_val!=l) & (batch_label==l) & (batch_smpw>0))
                # FP:
                FP[l] += np.sum((pred_val==l) & (batch_label!=l) & (batch_smpw>0))
    
                IoU[l] += TP[l]/float(TP[l]+FP[l]+FN[l])
                MA[l] += total_correct_class[l]/float(total_seen_class[l])
    
                if total_seen_class[l] == 0:
                    # Esa clase no esta presente en la nube
                    print('...Esa clase no esta presente en la nube...')
            
    
            # import pdb
            # pdb.set_trace()
    
            MA = np.nanmean(MA)
            MIoU = np.mean(IoU)
    
    
    
    
            # Ahora vamos a hacer lo mismo pero por bloques, que es lo que decia 
            # por ahi arriba. Asi que, iterativamente en cada nube del batch:
            for b in xrange(batch_label.shape[0]):
                
                # Voxelizamos la nube de puntos con una funcion especificada en 
                # pc_util.py de este mismo directorio:
                _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(aug_data[b,batch_smpw[b,:]>0,:], np.concatenate((np.expand_dims(batch_label[b,batch_smpw[b,:]>0],1),np.expand_dims(pred_val[b,batch_smpw[b,:]>0],1)),axis=1), res=0.02)
                # Aclaracion:
                # · _: "No lo tengo claro pero da igual que se sobreescribe aqui xD"
                # · uvlabel: Etiqueta del voxel
                # · _ (ultimo parametro): Numero/Indice del voxel.
                    
                # Actualizamos el numero total de VOXELES correctamente segmentados:
                total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
                
                # Actualizamos el numero total de VOXELES con prediccion hecha:
                total_seen_vox += np.sum(uvlabel[:,0]>0)
                
                # Ahora hacemos igual que antes un histograma y analizamos las cla-
                # ses que hay en ese VOXEL:
                tmp,_ = np.histogram(uvlabel[:,0],range(NUMERO_BINS))
                labelweights_vox += tmp
                for l in range(NUM_CLASSES):
                        # Actualizamos el numero total de clases vistas y las que
                        # fueron correctamente segmentadas:
                        total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
        
                        total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))
        
        
        # print('DEBUG')
        # print(loss_sum)
        # print(float(total_seen_vox))
        # print('DEBUG')
        
        # Mostramos en pantalla los resultados de la evaluacion:
        # try:
            # Perdidas promedio durante la evaluacion:
        log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        # except:
            # log_string('eval mean loss: DIVISION ENTRE 0!!! DEBUG 1')
            
        # try:
            # Precision en la clasificacion por voxeles:
        log_string('eval point accuracy vox: %f'% (total_correct_vox / float(total_seen_vox)))
        # except:
            # log_string('eval point accuracy vox: DIVISION ENTRE 0!!! DEBUG 2')
            
        # try:
            # Precision promedio en la clasificacion de los puntos por voxel:
        log_string('eval point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))))
        # except:
            # log_string('eval point avg class acc vox: DIVISION ENTRE 0!!! DEBUG 3')
        
        # try:
            # Precision en la evaluacion de cada punto (IMPORTANTE!!!!!!!):
        
        #######################################################################
        
        # RESULTADOS PAPER:
        
        # Overall accuracy (OA):
        OA = total_correct / float(total_seen)
        log_string('eval point accuracy: %f'% OA)
        
        # Mean Accuracy (MA):
        log_string('MA: %f'% (MA))
        
        # Mean Loss:
        Mean_Loss = loss_sum / float(num_batches)
        log_string('Mean Loss: %f'% Mean_Loss)
        
        # MIoU:
        log_string('MIoU: %f'% (MIoU))
        
        #######################################################################
        # except:
        #     log_string('eval point accuracy: DIVISION ENTRE 0!!! DEBUG 4')
        
        # En PointNet++ hacen tambien una evaluacion bajo unos parametros de cali-
        # bracion preestablecidos (no me interesa demasiado asi que paso un poco
        # por encima hasta la linea de guiones: "-------"):
        # labelweights_vox = labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
        # caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
        # caliweights = caliweights[0:NUM_CLASSES-1]
        
        # log_string('eval point calibrated average acc: %f' % (np.average(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6),weights=caliweights)))
        # per_class_str = 'vox based --------'
        # for l in range(1,NUM_CLASSES):
        #     # try:
        #     per_class_str += 'class %d weight: %f, acc: %f; ' % (l,labelweights_vox[l-1],total_correct_class[l]/float(total_seen_class[l]))
        #     log_string(per_class_str)
        #     # except:
        #     #     per_class_str += 'class %d weight: DIVISION ENTRE 0!!!, acc: DIVISION ENTRE 0!!! DEBUG 5'
        #     #     log_string(per_class_str)
        
        # -------------------------------------------------------------------------
        
        # Ahora lo que hago es abrir un fichero y guardo las estadisticas:
        
        ruta_ruptura = os.getcwd()
        os.chdir(RUTA_LOG_ABSOLUTA)
            
        try:
            if epoch == 0:
                f = open("%s_accuracies_y_losses_one_epoch.txt"%nombre_sesion, "w")
                f.write("ACCURACIES    LOSSES    EPOCH    BATCH    NUBE_ID\n")
                f.write(str(total_correct / float(total_seen))+'    '+str(loss_sum / float(num_batches))+'    '+str(epoch)+'    '+str(batch_idx)+'    '+str(ll)+'\n')
                f.close()
            else:
                f = open("%s_accuracies_y_losses_one_epoch.txt"%nombre_sesion, "a")
                f.write(str(total_correct / float(total_seen))+'    '+str(loss_sum / float(num_batches))+'    '+str(epoch)+'    '+str(batch_idx)+'    '+str(ll)+'\n')
                f.close()
        except:
            if epoch == 0:
                f = open("%s_accuracies_y_losses_one_epoch.txt"%nombre_sesion, "w")
                f.write("ACCURACIES    LOSSES    EPOCH    BATCH    NUBE_ID\n")
                f.write('error'+'    '+'error'+'    '+str(epoch)+'    '+str(batch_idx)+'    '+str(ll)+'\n')
                f.close()
            else:
                f = open("%s_accuracies_y_losses_one_epoch.txt"%nombre_sesion, "a")
                f.write('error'+'    '+'error'+'    '+str(epoch)+'    '+str(batch_idx)+'    '+str(ll)+'\n')
                f.close()
        
        
        
        # RESULTADOS PAPER:
        if epoch == 0:
            f = open("RESULTADOS_PAPER_%s_TEST.txt"%nombre_sesion, "w")
            f.write("Overall Accuracy (OA)    Mean Accuracy (MA)    Mean Loss    MIoU    Epoch    NUBE_ID\n")
            f.write(str(OA)+'    '+str(MA)+'    '+str(Mean_Loss)+'    '+str(MIoU)+'    '+str(epoch)+'    '+str(ll)+'\n')
            f.close()
        else:
            f = open("RESULTADOS_PAPER_%s_TEST.txt"%nombre_sesion, "a")
            f.write(str(OA)+'    '+str(MA)+'    '+str(Mean_Loss)+'    '+str(MIoU)+'    '+str(epoch)+'    '+str(ll)+'\n')
            f.close()
        
        
        
        
        
        
        
        
        
        os.chdir(ruta_ruptura)
        
    # Aumentamos el contador de epocas:
    EPOCH_CNT += 1
    
    
    
    contador_dataset_testeo = contador_dataset_testeo_original
    
    # El valor final que regresamos es la precision absoluta:
    try:
        return total_correct/float(total_seen)
    except:
        return 'error'



# Definimos ahora la funcion de evaluacion sobre NUBES COMPLETAS:
'''
COMENTARIOS DE LOS DESARROLLADORES:
# evaluate on whole scenes to generate numbers provided in the paper
'''
def eval_whole_scene_one_epoch(sess, ops, test_writer,epoch,RUTA_LOG_ABSOLUTA):
    """ ops: dict mapping from string to tf ops """
    
    # Esta primera parte va a ser practicamente igual a la funcion de evalua-
    # cion en nubes cortadas de antes.
    
    
    global EPOCH_CNT
    is_training = False
    
    # Indices de cada nube en el set de testeo 'whole scene':
    test_idxs = np.arange(0, len(TEST_DATASET_WHOLE_SCENE))
    
    # Numero de batches que tendremos en el set de testeo 'whole scene':
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    # Definimos estas variables que actualizaremos en un rato:
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    # Definimos estas variables tambien que haran referencia a las mismas can-
    # tidades pero medidas en cada "caja" (recordemos que PointNet++ hace esta 
    # evaluacion por bloques que recorta de la nube):
    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]
    
    # Mostramos en pantalla la fecha del momento actual y que nos encontramos 
    # en la fase de evaluacion de la epoca actual:
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION WHOLE SCENE----'%(epoch))

    # Creamos unos arrays de ceros que haran referencia a los pesos de cada la-
    # bel:
    labelweights = np.zeros(NUM_CLASSES+1)
    labelweights_vox = np.zeros(NUM_CLASSES+1)
    
    # Definimos esta variable booleana que hara referencia a si quedan nubes
    # por evaluar en el batch o no:
    is_continue_batch = False
    
    # Definimos unos arrays de ceros que nos servirán para rellenar los batches
    # finales que puedan quedar "medio incompletos" (rollo cuando la longitud 
    # del batch no sea multiplo entero del numero de batches y quede algo, como
    # por ejemplo, un array donde cada elemento es 3 y el último 2.1):
    extra_batch_data = np.zeros((0,NUM_POINT,3))
    extra_batch_label = np.zeros((0,NUM_POINT))
    extra_batch_smpw = np.zeros((0,NUM_POINT))
    
    # Iteramos sobre cada batch:
    for batch_idx in range(num_batches):
        
        # En el caso de que el batch final no este completo lo rellenamos con 
        # el batch "de ceros" que creamos ates:
        if not is_continue_batch:
            batch_data, batch_label, batch_smpw = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data,extra_batch_data),axis=0)
            batch_label = np.concatenate((batch_label,extra_batch_label),axis=0)
            batch_smpw = np.concatenate((batch_smpw,extra_batch_smpw),axis=0)
        
        # En el caso de que el batch este perfectamente completo:
        else:
            batch_data_tmp, batch_label_tmp, batch_smpw_tmp = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data,batch_data_tmp),axis=0)
            batch_label = np.concatenate((batch_label,batch_label_tmp),axis=0)
            batch_smpw = np.concatenate((batch_smpw,batch_smpw_tmp),axis=0)
        
        # En ambas partes de ese condicional de aqui arriba lo que estamos ha-
        # ciendo es cargar los datos de este batch de testeo 'whole scene'.

        
        # Si la cantidad de nubes en este batch es igual al tamanho especifica-
        # do por el usuario (BATCH_SIZE), entonces sabemos que puede haber mas
        # nubes sobre las que hacer evaluaciones, asi que hacemos unos condi-
        # cionales en los que recogemos esta idea:
            
        if batch_data.shape[0]<BATCH_SIZE:
            is_continue_batch = True
            continue
        elif batch_data.shape[0]==BATCH_SIZE:
            is_continue_batch = False
            extra_batch_data = np.zeros((0,NUM_POINT,3))
            extra_batch_label = np.zeros((0,NUM_POINT))
            extra_batch_smpw = np.zeros((0,NUM_POINT))
        else:
            is_continue_batch = False
            extra_batch_data = batch_data[BATCH_SIZE:,:,:]
            extra_batch_label = batch_label[BATCH_SIZE:,:]
            extra_batch_smpw = batch_smpw[BATCH_SIZE:,:]
            batch_data = batch_data[:BATCH_SIZE,:,:]
            batch_label = batch_label[:BATCH_SIZE,:]
            batch_smpw = batch_smpw[:BATCH_SIZE,:]
    
        # Se nota que esto fue un copypaste, lo que hacemos es tomar los datos
        # tal cual, sin someterlos a data augmentation:
        aug_data = batch_data
        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                    ops['labels_pl']: batch_label,
                    ops['smpws_pl']: batch_smpw,
                    ops['is_training_pl']: is_training}
        # Recordemos que es cada cosa:
        # · pointclouds_pl: Array con las nubes de puntos en el batch.
        # · labels_pl: Array donde cada elemento son las labels de cada nube
        #              del batch.
        # · smpws_pl: Array donde cada elemento son los pesos de cada nube del
        #             batch.
        # · is_training_pl: Variable que hace referencia a si estamos entrenan-
        #                   do o evaluando.
        
        # INTRODUCIMOS DATOS EN LA RED:
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                  ops['loss'], ops['pred']], feed_dict=feed_dict)
       
        # Escribimos el summary de esta fase de testeo 'whole scene':
        test_writer.add_summary(summary, step)
        
        # A PARTIR DE AQUI TODO LO QUE QUEDA DEL BUCLE ES COMPLETAMENTE ANALOGO
        # A LO QUE SE HACE EN LA DEFINICION DE eval_one_epoch():
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0)) 
        total_correct += correct
        total_seen += np.sum((batch_label>0) & (batch_smpw>0))
        loss_sum += loss_val
        NUMERO_BINS = NUM_CLASSES + 2
        tmp,_ = np.histogram(batch_label,range(NUMERO_BINS))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
    
        for b in xrange(batch_label.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(aug_data[b,batch_smpw[b,:]>0,:], np.concatenate((np.expand_dims(batch_label[b,batch_smpw[b,:]>0],1),np.expand_dims(pred_val[b,batch_smpw[b,:]>0],1)),axis=1), res=0.02)
            total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
            total_seen_vox += np.sum(uvlabel[:,0]>0)
            tmp,_ = np.histogram(uvlabel[:,0],range(NUMERO_BINS))
            labelweights_vox += tmp
            for l in range(NUM_CLASSES):
                    total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
                    total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

    try:
        log_string('eval whole scene mean loss: %f' % (loss_sum / float(num_batches)))
    except:
        log_string('eval whole scene mean loss: DIVISION ENTRE 0!!! DEBUG 6')
    try:
        log_string('eval whole scene point accuracy vox: %f'% (total_correct_vox / float(total_seen_vox)))
    except:
        log_string('eval whole scene point accuracy vox: DIVISION ENTRE 0!!! DEBUG 7')
    try:
        log_string('eval whole scene point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))))
    except:
        log_string('eval whole scene point avg class acc vox: DIVISION ENTRE 0!!! DEBUG 8')
    try:
        log_string('eval whole scene point accuracy: %f'% (total_correct / float(total_seen)))
    except:
        log_string('eval whole scene point accuracy: DIVISION ENTRE 0!!! DEBUG 9')
    try:
        log_string('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    except:
        log_string('eval whole scene point avg class acc: DIVISION ENTRE 0!!! DEBUG 10')
    
    
    # Ahora lo que hago es abrir un fichero y guardo las estadisticas:
    try:
        if EPOCH_CNT == 0:
            f = open("%s_accuracies_y_losses_one_epoch_WHOLE_SCENE.txt"%nombre_sesion, "w")
            f.write("ACCURACIES    LOSSES    EPOCH\n")
            f.write(str(total_correct / float(total_seen))+'    '+str(loss_sum / float(num_batches))+'    '+str(epoch)+'\n')
            f.close()
        else:
            f = open("%s_accuracies_y_losses_one_epoch_WHOLE_SCENE.txt"%nombre_sesion, "a")
            f.write(str(total_correct / float(total_seen))+'    '+str(loss_sum / float(num_batches))+'    '+str(epoch)+'\n')
            f.close()
    except:
        if EPOCH_CNT == 0:
            f = open("%s_accuracies_y_losses_one_epoch_WHOLE_SCENE.txt"%nombre_sesion, "w")
            f.write("ACCURACIES    LOSSES    EPOCH\n")
            f.write('error'+'    '+'error'+'    '+str(epoch)+'\n')
            f.close()
        else:
            f = open("%s_accuracies_y_losses_one_epoch_WHOLE_SCENE.txt"%nombre_sesion, "a")
            f.write('error'+'    '+'error'+'    '+str(epoch)+'\n')
            f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    labelweights = labelweights[1:].astype(np.float32)/np.sum(labelweights[1:].astype(np.float32))
    labelweights_vox = labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
    caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    caliweights = caliweights[0:NUM_CLASSES-1]
    caliacc = np.average(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6),weights=caliweights)
    log_string('eval whole scene point calibrated average acc vox: %f' % caliacc)

    per_class_str = 'vox based --------'
    for l in range(1,NUM_CLASSES):
        per_class_str += 'class %d weight: %f, acc: %f; ' % (l,labelweights_vox[l-1],total_correct_class_vox[l]/float(total_seen_class_vox[l]))
    log_string(per_class_str)


        

    EPOCH_CNT += 1
    
    '''
    LOS DESARROLLADORES LO TENIAN ASI:    
    return caliacc

    PERO YO ME VOY A QUEDAR CON LA OTRA PRECISION MEJOR, QUE A MI NO ME INTERE-
    SA SUS METRICAS.
    '''

    # El valor final que regresamos es la precision absoluta:
    try:
        return total_correct/float(total_seen)
    except:
        return 'error'


















def eval_one_epoch_PREDICCION_NUBE_REAL(sess, ops, test_writer,epoch,RUTA_LOG_ABSOLUTA):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    
    
    for ll in range(NUMERO_NUBES_PREDICCION):
    
        global contador_dataset_testeo
        
        PREDICT_DATASET = scannet_dataset.ScannetDataset(root=PREDICT_DATA_PATH, npoints=NUM_POINT, split='prediccion',LINO=LINO,NUM_CLASSES=NUM_CLASSES,downsampleamos=False,contador_dataset=0,tamanho_batch = 1,tipo_de_nubes=tipo_de_nubes,archivo='Nube_artificial_0.npy',algunas_clases_fusionadas=False)
        # contador_dataset_testeo += BATCH_SIZE
    
    
    
    
    
        is_training = False
    
    
    # TEST_DATASET = scannet_dataset.ScannetDataset(root=TEST_DATA_PATH, npoints=NUM_POINT, split='test',LINO=LINO,NUM_CLASSES=NUM_CLASSES,downsampleamos=True, contador_dataset=contador_dataset_testeo, tamanho_batch=BATCH_SIZE)
        
        # TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=TEST_DATA_PATH, npoints=NUM_POINT, split='test',LINO=LINO,NUM_CLASSES=NUM_CLASSES,downsampleamos=True)
        
        
        
        # Indices de cada nube en el set de testeo:
        test_idxs = np.arange(0, len(PREDICT_DATASET))
        
        # Numero de batches que tendremos en el set de testeo:
        num_batches = len(PREDICT_DATASET)/BATCH_SIZE
    
    
        # log_string(str(contador_dataset_testeo))
        # log_string(str(BATCH_SIZE))
        # log_string('holaaaaaaaaaaaaaaaaaaaaaaaaaa')
    
    
    
        # Definimos estas variables que actualizaremos en un rato:
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
        # Definimos estas variables tambien que haran referencia a las mismas can-
        # tidades pero medidas en cada "caja" (recordemos que PointNet++ hace esta 
        # evaluacion por bloques que recorta de la nube):
        total_correct_vox = 0
        total_seen_vox = 0
        total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
        total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]
        
        # Mostramos en pantalla la fecha del momento actual y que nos encontramos 
        # en la fase de evaluacion de la epoca actual:
        log_string(str(datetime.now()))
        log_string('---- EPOCH %03d PREDICCION NUBE SIN ETIQUETAR ----'%(epoch))
    
        # Creamos unos arrays de ceros que haran referencia a los pesos de cada la-
        # bel:
        labelweights = np.zeros(NUM_CLASSES+1)
        labelweights_vox = np.zeros(NUM_CLASSES+1)
        
        # Iteramos para cada batch de nubes en el set de testeo:
        for batch_idx in range(num_batches):
           
    
            # Indices del batch inicial y final:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            
            
            
            
            # Cargamos los datos de este batch de testeo:
            batch_data, batch_label, batch_smpw, indices_puntos_seleccionados = get_batch(PREDICT_DATASET, test_idxs, start_idx, end_idx)
    
            # Hacemos data augmentation rotando las nubes de puntos de cada batch
            # mediante la funcion rotate_point_cloud_z del modulo provider:
            aug_data = provider.rotate_point_cloud_z(batch_data)
    
            # Introduciremos en la red los datos en forma de diccionario con los 
            # datos aumentados de cada batch, los indices de cada batch, los pesos 
            # la variable switch de entrenamiento (booleana):        
            feed_dict = {ops['pointclouds_pl']: aug_data,
                         ops['labels_pl']: batch_label,
                         ops['smpws_pl']: batch_smpw,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            # Recordemos que es cada cosa:
            # · merged: Los summaries todos fusionados.
            # · step: El batch en el que nos encontramos.
            # · loss: Las perdidas en esa fase del entrenamiento.
            # . pred: El array de predicciones en esa fase del entrenamiento.
            # · feed_dict: Los datos con los que alimentamos la red en forma de 
            #             diccionario.
            
            # Escribimos el summary de esta fase de testeo:
            test_writer.add_summary(summary, step)
            
            # Cogemos como valores de prediccion aquellos que se repitan mas:
            pred_val = np.argmax(pred_val, 2) # BxN
            # print(np.count_nonzero(batch_label))
            
            
            #######################################################################
            #                        PARON PARA VISUALIZAR
            # Voy a guardar aqui los arrays de prediccion y la propia nube de pun-
            # tos para dibujarla externamente y ver que pasa:
            
            ruta_ruptura = os.getcwd()
            
            os.chdir(RUTA_LOG_ABSOLUTA)
            
            with open("PREDICCION_NUBE_REAL_epoca_%i.npy"%epoch, 'wb') as f:    
                np.save(f, aug_data)
                np.save(f,pred_val)
                
                
            with open('INDICES_PUNTOS_SELECCIONADOS_epoca_%i.npy'%epoch,'wb') as f:
                np.save(f,indices_puntos_seleccionados)
                
                
                
            # Para cargar esos arrays haríamos así:
                
            # import numpy as np
            # import open3d as o3d
            # import os
            # os.chdir('/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_LUNES')
            # with open("PREDICCION_NUBE_REAL_epoca_%i.npy"%285, 'rb') as f:
            #     aug_data = np.load(f)[0]
            #     pred_val = np.load(f)[0]
                    
            # indices_arboles = np.where(pred_val == 0)
            # indices_DTM = np.where(pred_val == 1)
            
            # colores = np.zeros(shape=aug_data.shape)
            # colores[indices_arboles] = [0,1,0]
            # colores[indices_DTM] = [1,0,0]
            
            # import open3d as o3d
            # nube_clasificada = o3d.geometry.PointCloud()
            # nube_clasificada.points = o3d.utility.Vector3dVector(aug_data)
            # nube_clasificada.colors = o3d.utility.Vector3dVector(colores)
            # o3d.visualization.draw(nube_clasificada)
            
            # # Volvemos a donde estabamos:
            os.chdir(ruta_ruptura)
            
            #######################################################################
    
        
    
            # A PARTIR DE AQUI TODO LO QUE QUEDA DEL BUCLE ES COMPLETAMENTE ANALOGO
            # A LO QUE SE HACE EN LA DEFINICION DE eval_one_epoch():
            correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0)) 
            total_correct += correct
            total_seen += np.sum((batch_label>0) & (batch_smpw>0))
            loss_sum += loss_val
            NUMERO_BINS = NUM_CLASSES + 2
            tmp,_ = np.histogram(batch_label,range(NUMERO_BINS))
            labelweights += tmp
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
                total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
        
            for b in xrange(batch_label.shape[0]):
                _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(aug_data[b,batch_smpw[b,:]>0,:], np.concatenate((np.expand_dims(batch_label[b,batch_smpw[b,:]>0],1),np.expand_dims(pred_val[b,batch_smpw[b,:]>0],1)),axis=1), res=0.02)
                total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
                total_seen_vox += np.sum(uvlabel[:,0]>0)
                tmp,_ = np.histogram(uvlabel[:,0],range(NUMERO_BINS))
                labelweights_vox += tmp
                for l in range(NUM_CLASSES):
                        total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
                        total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))
    
        log_string('PREDICCION mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('PREDICCION point accuracy vox: %f'% (total_correct_vox / float(total_seen_vox)))
        log_string('PREDICCION point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))))
        log_string('PREDICCION point accuracy: %f'% (total_correct / float(total_seen)))
        log_string('PREDICCION point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
        
        
        
        ruta_ruptura = os.getcwd()
        
        os.chdir(RUTA_LOG_ABSOLUTA)
        
        # Ahora lo que hago es abrir un fichero y guardo las estadisticas:
        if "%s_accuracies_y_losses_NUBE_REAL.txt"%nombre_sesion not in os.listdir(os.getcwd()):
            f = open("%s_accuracies_y_losses_NUBE_REAL.txt"%nombre_sesion, "w")
            f.write("ACCURACIES    LOSSES    EPOCH\n")
            f.write(str(total_correct / float(total_seen))+'    '+str(loss_sum / float(num_batches))+'    '+str(epoch)+'\n')
            f.close()
        else:
            f = open("%s_accuracies_y_losses_NUBE_REAL.txt"%nombre_sesion, "a")
            f.write(str(total_correct / float(total_seen))+'    '+str(loss_sum / float(num_batches))+'    '+str(epoch)+'\n')
            f.close()
        
        os.chdir(ruta_ruptura)
    
    # Aumentamos el contador de epocas:
    EPOCH_CNT += 1
    
    
    
    # contador_dataset_testeo = contador_dataset_testeo_original
    
    return





























if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
    
    
    
    
    # Aqui ya acabo de entrenar, por lo que voy a leer los datos expulsados y
    # montar unas graficas, pero eso lo haremos en otro script.
        
    
    
    
