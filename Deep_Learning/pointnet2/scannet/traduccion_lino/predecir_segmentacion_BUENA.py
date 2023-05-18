#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:51:20 2022

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


# Numero total de epocas sobre las cuales entrenar el modelo:
NUMERO_DE_EPOCAS = 20000

# Va a haber dos downsampleos, uno que es el que se hace en el propio codigo de
# Pointnet++ (NUMERO_PUNTOS_VOXEL_PN) y otro que hago yo antes de meter las nu-
# bes en la red (NUMERO_PUNTOS_VOXEL_LINO):
NUMERO_PUNTOS_VOXEL_PN = 8192
NUMERO_PUNTOS_VOXEL_LINO = 200000

# Numero de nubes en las que se centrara el entrenamiento por cada vez (batches
# mas pequenhos hara que tarde y batches mas grandes requeriran mas memoria):
TAMANHO_BATCH = 1

# Modelo que vamos a emplear (ya que estamos con segmentacion pues usaremos el
# de segmentacion):
MODELO = 'pointnet2_sem_seg'

# Numero de clases que tendran las nubes con las que vamos a entrenar y testear
# Ejemplo: nubes con terreno y arboles (2 clases).
NUM_CLASSES = 2

# Por algun motivo especial interesa poner el nombre del usuario del PC (sale 
# en el log file pero bueno...)
HOSTNAME = socket.gethostname()


MODEL_PATH = '/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_autopista_5_capas_seccion_transversal'

# Cargamos el nombre del modelo ya entrenado:
nombre_del_modelo = '20_epoch_model.ckpt'

#------------------------------------------------------------------------------
#???

# ARREGLAMOS ALGUNOS PATHS IMPORTANTES (y terminamos de importar modulos):
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2'
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append('/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/models')
sys.path.append('/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/utils')

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
parser.add_argument('--log_dir', default='log_%s'%MODEL_PATH, help='Log dir [default: log_%s]'%MODEL_PATH)
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
LOG_DIR = MODEL_PATH


# Creo una variable del path absoluto del LOG_DIR:
ruta_actualisima = os.getcwd()
os.chdir(LOG_DIR)
RUTA_LOG_ABSOLUTA = os.getcwd()
os.chdir(ruta_actualisima)


# # Se intentaran copiar unos archivos a modo backup, si falla que no cunda el
# # panico que no es muy importante:
# os.system('cp %s %s' % (MODEL_FILE, RUTA_LOG_ABSOLUTA)) # bkp of model def
# os.system('cp train.py %s' % (RUTA_LOG_ABSOLUTA)) # bkp of train procedure

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
    # Shapenet official train/test split
    DATA_PATH = os.path.join(ROOT_DIR,'data','scannet_data_pointnet2')
    TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train',LINO=LINO,NUM_CLASSES=NUM_CLASSES)
    TEST_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test',LINO=LINO,NUM_CLASSES=NUM_CLASSES)
    TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test',LINO=LINO,NUM_CLASSES=NUM_CLASSES)
else:
    # Usamos mis propios datos:
    PREDICT_DATA_PATH = '/home/lino/Documentos/programas_pruebas_varias/segmentacion_python/segmentacion_bosques/aumentacion_de_datos/Nubes_artificiales_generadas/Nubes_artificiales_personalizadas_2022_7_4___9:40:20/Nube_artificial_0/numpy_arrays'    
    nombre_archivo_a_segmentar = 'Nube_artificial_0.npy'

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
            # print(is_training_pl)
            
        #     # Notas de los desarrolladores de PN++:
        #     '''
        #     # Note the global_step=batch parameter to minimize. 
        #     # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        #     '''
            
            # Creamos dos variables de Tensorflow, una asociada a un batch ge-
            # nerico y otra al bn_decay.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
        #     # Anhadimos el bn_decay como un escalar a la tabla summary de TF:
        #     tf.summary.scalar('bn_decay', bn_decay)

        #     print("--- Get model and loss")
        #     # Vamos a obtener el modelo y las perdidas.
            
            # Sustraemos dos arrays, uno con las predicciones y otro un poco
            # mas especial que explico justo despues:
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            # end_points es un diccionario con dos elementos, uno que es la nu-
            # de puntos que introducimos y otro con las features sustraidas, al
            # final del modelo (ver pointnet2_sem_seg.py).
            
            # Obtenemos las perdidas como forma de array:
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
        #     # Anhadimos las perdidas como un escalar a la tabla summary de TF:
        #     tf.summary.scalar('loss', loss)

        #     # Comparamos dos tensores, uno con las predicciones mas probables
        #     # (de ahi lo del maximo) y otro con las etiquetas reales:
        #     # (Ojo a las explicaciones de justo despues)
        #     correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
        #     # tf.argmax(array, indice_dimension) --->  Devuelve un array con 
        #     #                                          los indices de los ele-
        #     #                                          mentos maximos del se-
        #     #                                          gundo atributo 'indice_dimension'
            
        #     # tf.to_int64('tensor en placeholder') ---> Devuelve el mismo ten-
        #     #                                           sor pero como int64
            
        #     # Sacamos el valor de la precision en este batch:
        #     accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
        #     # Anhadimos las precision como un escalar a la tabla summary de TF:
        #     tf.summary.scalar('accuracy', accuracy)

        #     print("--- Get training operator")
        #     # Vamos a obtener el operador del entrenamiento
            
            # Cogemos el learning rate segun la funcion previamente explicada:
            learning_rate = get_learning_rate(batch)
        #     # Anhadimos el learning_rate como un escalar a la tabla summary de TF:
        #     tf.summary.scalar('learning_rate', learning_rate)
            
            # Especificamos el optimizador que vamos a usar de la coleccion de TF:
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
        #     # Nota de los desarrolladores de PN++:
        #     '''
        #     # Add ops to save and restore all the variables.
        #     '''
        #     # Creamos un saver, que sirve para crear checkpoints durante el en-
        #     # trenamiento y asi poder entrenar/testear/etc el modelo desde una
        #     # epoca concreta:
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
        
        
        # saver = tf.train.import_meta_graph(MODEL_PATH+'/'+nombre_del_modelo+'.meta')
        
        
        # Restore variables from disk.
        # saver.restore(sess, MODEL_PATH+'/'+nombre_del_modelo)
        saver.restore(sess, os.path.join(MODEL_PATH, nombre_del_modelo))

        
        log_string("Modelo restaurado.")
        
        
        
        
        # Fusionamos todos los 'summaries' recolectados hasta ahora:
        # merged = tf.summary.merge_all()
        
        
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
               # 'merged': merged,
               'step': batch,
               'end_points': end_points}

        
        # PEDIMOS HACER UNA EVALUACION AL MODELO YA ENTRENADO!!!
        eval_one_epoch(sess, ops, test_writer,RUTA_LOG_ABSOLUTA)













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
        ps,seg,smpw, indices = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
        indices_puntos_seleccionados[i,:] = indices
    return batch_data, batch_label, batch_smpw, indices_puntos_seleccionados







def eval_one_epoch(sess, ops, test_writer,RUTA_LOG_ABSOLUTA):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    
        
    PREDICT_DATASET = scannet_dataset.ScannetDataset(root=PREDICT_DATA_PATH, npoints=NUM_POINT, split='prediccion',LINO=LINO,NUM_CLASSES=NUM_CLASSES,downsampleamos=False,contador_dataset=0,tamanho_batch = BATCH_SIZE, archivo=nombre_archivo_a_segmentar)


    is_training = False
    
    # Indices de cada nube en el set de testeo:
    test_idxs = np.arange(0, len(PREDICT_DATASET))
    
    # Numero de batches que tendremos en el set de testeo:
    num_batches = len(PREDICT_DATASET)/BATCH_SIZE



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
    log_string('---- PREDICCIONNNNNNN ----')

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


        #######################################################################
        # PARON PARA VISUALIZAR
        # Para cargar esos arrays haríamos así:
            
            
        
        ruta_ruptura = os.getcwd()
        
        os.chdir(RUTA_LOG_ABSOLUTA)
        
        with open("test_entrada_datos_en_modelo_prediccion.npy", 'wb') as f:    
            np.save(f, aug_data)
            np.save(f,batch_label)
            
        # import numpy as np
        # import open3d as o3d
        # import os
        # os.chdir('/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_dataset_grande')
        # with open("test_entrada_datos_en_modelo_prediccion.npy", 'rb') as f:
        #     aug_data = np.load(f)[0]
        #     batch_label = np.load(f)[0]
                
        # indices_arboles = np.where(batch_label == 0)
        # indices_DTM = np.where(batch_label == 1)
        
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


        # Introduciremos en la red los datos en forma de diccionario con los 
        # datos aumentados de cada batch, los indices de cada batch, los pesos 
        # la variable switch de entrenamiento (booleana):        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        
        
        step, loss_val, pred_val = sess.run([ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        
        # Recordemos que es cada cosa:
        # · merged: Los summaries todos fusionados.
        # · step: El batch en el que nos encontramos.
        # · loss: Las perdidas en esa fase del entrenamiento.
        # . pred: El array de predicciones en esa fase del entrenamiento.
        # · feed_dict: Los datos con los que alimentamos la red en forma de 
        #             diccionario.
        
        # Escribimos el summary de esta fase de testeo:
        # test_writer.add_summary(summary, step)
        
        # Cogemos como valores de prediccion aquellos que se repitan mas:
        pred_val = np.argmax(pred_val, 2) # BxN
        # print(np.count_nonzero(batch_label))
        
        
        #######################################################################
        #                        PARON PARA VISUALIZAR
        # Voy a guardar aqui los arrays de prediccion y la propia nube de pun-
        # tos para dibujarla externamente y ver que pasa:
        
        ruta_ruptura = os.getcwd()
        
        os.chdir(RUTA_LOG_ABSOLUTA)
        
        with open("prediccion_nube.npy", 'wb') as f:    
            np.save(f, aug_data)
            np.save(f,pred_val)
            
        # # Para cargar esos arrays haríamos así:
            
        # import numpy as np
        # import open3d as o3d
        # import os
        # os.chdir('/home/lino/Documentos/programas_pruebas_varias/PointNet/pointnet2/log_dataset_grande')
        # with open("prediccion_nube.npy", 'rb') as f:
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


    return
    
    
    
    
    

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
    
    
    
    
    
    
    
    
    
    
