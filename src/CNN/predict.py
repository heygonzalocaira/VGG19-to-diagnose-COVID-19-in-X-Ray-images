import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras



longitud, altura = 224, 224
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'

cnn =tf.keras.models.load_model(modelo)
#cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x) ##[[1,0]]
  result = array[0] ##[1,0]
  print(result)
  answer = np.argmax(result) # 0
  if answer == 0:
    print(file +" ----------- Lungs with Covid")
  elif answer == 1:
    print(file +" ----------- Normal lungs")
  return answer

rootDir = './test'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Folder found: %s' % dirName)
    for fname in fileList:
        predict(rootDir+"/" +str(fname))