from picamera2 import Picamera2, Preview
import time
import keyboard
import os


dispW=320
dispH=240

picam2=Picamera2()
picam2.preview_configuration.main.size=(dispW,dispH)
picam2.resolution=(dispW, dispH)
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start_preview(True)
picam2.start()

Label1 = "congorraylentes"
path_imagen = "/home/rpi/examples/lite/examples/image_classification/raspberry_pi/testeo/" + Label1 + "/"
if not os.path.exists(path_imagen): #Si el directorio no existe, lo crea
    os.mkdir(path_imagen)

def count_jpg_files(directory):
    return len([file for file in os.listdir(directory) if file.endswith('.jpg')])

number_of_jpg_files = count_jpg_files(path_imagen)

a = number_of_jpg_files
print("Actualmente hay "+str(a)+" imágenes en ese folder, se seguirán agregando.")

time.sleep(20) #Primero espera 20 seg por si acaso

while True:
    time.sleep(0.3) #Toma fotos cada 0.3 segundos
    a += 1
    pathimagen= path_imagen + Label1 + "_" + str(a) + ".jpg"
    picam2.capture_file(pathimagen)
