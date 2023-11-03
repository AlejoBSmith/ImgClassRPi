# Esto es para el live classification en la RPi, en la PC se hace con el loadtflitemodelinferencia.py

import argparse
import sys
import time
import cv2
from picamera2 import Picamera2
import numpy as np
import tensorflow as tf
from PIL import Image

dispW=320
dispH=240

picam2=Picamera2()
picam2.preview_configuration.main.size=(dispW,dispH)
picam2.preview_configuration.main.format="RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Carga el tflite y le pone los valores de los pesos y bias al interpretador
interpreter = tf.lite.Interpreter(model_path="desktop.tflite")
interpreter.allocate_tensors()

# Obtiene la información de los metadatos
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the image
while True:
  image = picam2.capture_array()
  cv2.imshow('image_classification', image)
  image = np.rot90(image)
  input_array = np.asarray(image).astype(np.float32) #Yo creo que esto se está ejecutando más lento por estar usando este tipo de dato
  input_array = np.expand_dims(input_array, axis=0) #Esto es para agregar la dimensión de los batches, es decir, cuánta imágenes se le alimentan a la vez

  # Es posible que si no se definieron las dimensiones de la imagen correctamente, aquí vaya a salir error.
  # Aquí se imprime cuáles dimensiones espera el modelo y cuáles se les está enviando
  # Aquí hay que hacer una rotación, pero no tengo idea por qué entra girada 90 grados?
  # print("Model's expected input shape:", input_details[0]['shape'])
  # print("Your input shape:", input_array.shape)
  # input_array = np.asarray(image).astype(np.float32)  # Shape should match model's input shape
  # input_array = np.expand_dims(input_array, axis=0)
  # print("Your input shape:", input_array.shape)

  interpreter.set_tensor(input_details[0]['index'], input_array)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])

  print("Output data:", output_data) #Estos prints solo están aquí por debugging, mientras menos se tengan, más rápido se ejecuta!

  def softmax(x):
      exp_x = np.exp(x - np.max(x))
      return exp_x / exp_x.sum(axis=-1, keepdims=True)

  output_probabilities = softmax(output_data)

  # Get the index of the maximum value
  max_index = np.argmax(output_probabilities)

  # Create a zero vector of the same shape as the output
  one_hot_output = np.zeros(output_probabilities.shape[-1])

  # Set the entry at the max_index to 1
  one_hot_output[max_index] = 1

  print("One-hot output:", one_hot_output)
  class_names = ["congorraylentes", "controlaire", "controlps4", "perfume"]
  max_index = np.argmax(output_probabilities)
  predicted_class = class_names[max_index]

  print("Predicted Class:", predicted_class)
    # Stop the program if the ESC key is pressed.
  if cv2.waitKey(1) == 27:
    break

picam2.stop()
cv2.destroyAllWindows()
