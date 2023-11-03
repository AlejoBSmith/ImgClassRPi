import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors information
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tipoimagen="controlps4"
numeroimagen=200
# Open the image
image = Image.open("testeo/"+tipoimagen+"/"+tipoimagen+"_"+str(numeroimagen)+".jpg")
input_array = np.asarray(image).astype(np.float32)  # Shape should match model's input shape

# Add a batch dimension if your model expects it (uncomment line below if needed)
input_array = np.expand_dims(input_array, axis=0)

# Es posible que si no se definieron las dimensiones de la imagen correctamente, aquí vaya a salir error.
# Aquí se imprime cuáles dimensiones espera el modelo y cuáles se les está enviando
# Aquí hay que hacer una rotación, pero no tengo idea por qué entra girada 90 grados?
print("Model's expected input shape:", input_details[0]['shape'])
print("Your input shape:", input_array.shape)
image = image.rotate(90, expand=True)
input_array = np.asarray(image).astype(np.float32)  # Shape should match model's input shape
input_array = np.expand_dims(input_array, axis=0)
print("Your input shape:", input_array.shape)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_array)

# Run inference
interpreter.invoke()

# Get output data
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Output data:", output_data)

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