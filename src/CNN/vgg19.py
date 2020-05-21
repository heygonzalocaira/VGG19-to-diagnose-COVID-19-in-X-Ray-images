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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

K.clear_session()

data_entrenamiento = './dataset/Train' #x_train
data_validacion = './dataset/Validation' # y _train




"""
Parameters
"""

tamano_filtro = (3, 3)
tamano_pool = (2, 2)
longitud, altura = 224, 224


epocas=50
batch_size = 7 #10
pasos = 10
#validation_steps = 300
clases = 2
lr = 0.0001


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



inputs = keras.Input(shape=(altura,longitud,3))
#block 1
x = layers.Conv2D(64,tamano_filtro,padding="same",activation="relu",name='b1_conv1')(inputs)
x = layers.Conv2D(64,tamano_filtro,padding="same",activation="relu",name='b1_conv2')(x)
x = layers.MaxPooling2D(tamano_pool,(2,2), name='b1_pool')(x)
#block 2
x = layers.Conv2D(128,tamano_filtro,padding='same',activation='relu',name='b2_conv1')(x)
x = layers.Conv2D(128,tamano_filtro,padding="same",activation="relu",name='b2_conv2')(x)
x = layers.MaxPooling2D(tamano_pool,(2,2), name='b2_pool')(x)
#block 3
x = layers.Conv2D(256,tamano_filtro,padding='same',activation='relu',name='b3_conv1')(x)
x = layers.Conv2D(256,tamano_filtro,padding="same",activation="relu",name='b3_conv2')(x)
x = layers.Conv2D(256,tamano_filtro,padding='same',activation='relu',name='b3_conv3')(x)
x = layers.Conv2D(256,tamano_filtro,padding="same",activation="relu",name='b3_conv4')(x)
x = layers.MaxPooling2D(tamano_pool,(2,2), name='b3_pool')(x)
#block 4
x = layers.Conv2D(512,tamano_filtro,padding='same',activation='relu',name='b4_conv1')(x)
x = layers.Conv2D(512,tamano_filtro,padding="same",activation="relu",name='b4_conv2')(x)
x = layers.Conv2D(512,tamano_filtro,padding='same',activation='relu',name='b4_conv3')(x)
x = layers.Conv2D(512,tamano_filtro,padding="same",activation="relu",name='b4_conv4')(x)
x = layers.MaxPooling2D(tamano_pool,(2,2), name='b4_pool')(x)
#block 5
x = layers.Conv2D(512,tamano_filtro,padding='same',activation='relu',name='b5_conv1')(x)
x = layers.Conv2D(512,tamano_filtro,padding="same",activation="relu",name='b5_conv2')(x)
x = layers.Conv2D(512,tamano_filtro,padding='same',activation='relu',name='b5_conv3')(x)
x = layers.Conv2D(512,tamano_filtro,padding="same",activation="relu",name='b5_conv4')(x)
x = layers.MaxPooling2D(tamano_pool,(2,2), name='b5_pool')(x)

#NN
x = Flatten(name='flatten')(x)
x = layers.Dense(4096, activation="relu", name='fc1')(x)
x = layers.Dense(4096, activation="relu", name='fc2')(x)
outputs = layers.Dense(clases, activation="softmax", name='predictions')(x)

#cnn.add(layers.GlobalMaxPooling2D())

model = keras.Model(inputs=inputs, outputs=outputs, name='vgg19')
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, name='Adam'),  
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.fit_generator(
    entrenamiento_generador,#x_train
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,#y_train
    #validation_steps=validation_steps)
)
# Guardar el Model
target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./modelo/modelo.h5')
model.save_weights('./modelo/pesos.h5')
