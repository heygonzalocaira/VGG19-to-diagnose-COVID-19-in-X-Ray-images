import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K


K.clear_session()

data_entrenamiento = './dataset/Train'
data_validacion = './dataset/Validation'




"""
Parameters
"""
epocas=20
longitud, altura = 224, 224
batch_size = 32
pasos = 1000
validation_steps = 300
tamano_filtro = (3, 3)
tamano_pool = (2, 2)
clases = 2
lr = 0.0004


entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')


cnn = keras.Sequential()
cnn.add(keras.Input(shape=(altura,longitud,3)))
#block 1
cnn.add(layers.Conv2D(64,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(64,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.MaxPooling2D(tamano_pool,(2,2)))
#block 2
cnn.add(layers.Conv2D(128,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(128,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.MaxPooling2D(tamano_pool,(2,2)))
#block 3
cnn.add(layers.Conv2D(256,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(256,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(256,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(256,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.MaxPooling2D(tamano_pool,(2,2)))
#block 4
cnn.add(layers.Conv2D(512,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(512,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(512,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(512,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.MaxPooling2D(tamano_pool,(2,2)))
#block 5
cnn.add(layers.Conv2D(512,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(512,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(512,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.Conv2D(512,tamano_filtro,padding="same",activation="relu"))
cnn.add(layers.MaxPooling2D(tamano_pool,(2,2)))

cnn.add(Flatten(name='flatten'))
cnn.add(layers.Dense(4096, activation="relu", name='fc1'))
cnn.add(layers.Dense(4096, activation="relu", name='fc2'))
x = cnn.add(layers.Dense(clases, activation="softmax", name='predictions'))

#cnn.add(layers.GlobalMaxPooling2D())


cnn.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['sparse_categorical_accuracy'])
print('# Fit model on training data')
cnn.fit(entrenamiento_generador,validacion_generador,
        steps_per_epoch=pasos,
        epochs=epocas,
        #validation_data=validacion_generador,
        validation_steps=validation_steps)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
#print("(UwU")