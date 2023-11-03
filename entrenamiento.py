import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os

from tensorflow import keras
from keras import layers
from keras.applications import EfficientNetB0

import pathlib

# Aquí hay tres archivos para el proceso de entrenamiento y porto a la RPi
# El primero es el de entrenamiento, en el cual se cargan los datos, se entrena y se exporta el modelo.tflite
# El segundo es para hacer las inferencias usando el .tflite que se generó y verificar que sirve bien
# El tercero es el que escribe los metadatos. Sin esos metadatos, el código de la RPi no sirve.
# Para recordar, los metadatos es la información del modelo y justamente donde están los valores de salida
data_dir = "testeo/"
data_dir = pathlib.Path(data_dir).with_suffix('')

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
batch_size = 32
img_height = 320
img_width = 240

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# El autotune no es obligatorio pero sirve para agarrar el tamaño del búffer óptimo basado en el sitema.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

num_classes=len(class_names)

# Envés de definir una red a lo loco, mejor es agarrar una estructura de red existente; funciona mejor.
# Se usa el EfficienNetB0 como punto de partida, la cual ya está grabada en Keras
# Se cargan los pesos y los biases de la red ya entrenada
# Load EfficientNetB0 with pre-trained ImageNet weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Se congela para hacer transfer learning. O sea, se agarra una red preexistente y se fine-tunea para los datos
# específicos que se tiene. Para eso, se usa la EfficientNetB0 tal y como es, pero se le agregan dos capas al final
# que son las que se entrenan. Las capas anteriores no se tocan (recordando que las capas iniciales son las que aprenden
# los detalles más básicos como color, bordes, etc)
# Freeze the base model
base_model.trainable = False

# Para hacer el modelo custom (o, para hacer el transfer learning) se le agregan capas adicionales a la efficientnet
# Aquí el data_autmentation se agrega como una capa inicial para modificar todas las imágenes de entrada antes de hacer
# el entrenamiento. Estas imágenes solo se usan para el entrenamiento y NO para la inferencia. Esto es porque
# en Keras, las funciones "Random" no se consideran para el entrenamiento (eso está built-in en el mismo Keras)
# Create the model
model = tf.keras.Sequential([
  data_augmentation,
  base_model,
  layers.GlobalAveragePooling2D(),
  layers.Dense(num_classes, activation='softmax')  # Use 'softmax' for multi-class classification
])

# IMPORTANTE: Se entrena igual, solo que aquí hay que considerar cambiar de categorical_crossentropy a
# SparseCategoricalCrossentropy porque la función image_dataset_from_directory crea los labels como valores int
# es decir, la clase1 es 1, clase 2 es 2, etcétera. Para usar categorical_crossentropy, los labels tienen que
# estar en el formato [0, 1, 0, 0, 0]. Si se usa categorical_crossentropy, va a tirar un valor la inferencia,
# pero va a estar mal!
# Compile the model
model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

model.summary()

# Train the model
epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

nombrecarpeta='modeloentrenado'
model.save(nombrecarpeta, overwrite=True, save_format="tf") #Esto es para hacer el SavedModel, o sea, el original. Esto crea es una CARPETA, no un .tflite

# Convertir a tensorflow lite
converter = tf.lite.TFLiteConverter.from_saved_model(nombrecarpeta)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops = True
tflite_model = converter.convert()
# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model