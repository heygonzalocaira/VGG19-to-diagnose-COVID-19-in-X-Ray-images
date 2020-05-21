import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 224, 224
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Lungs with Covid")
  elif answer == 1:
    print("Normal lungs")
  return answer

file = input("Name of file : ")
predict(file)