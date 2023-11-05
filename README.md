# ImgClassRPi
Image classification for Raspberry Pi using tensorflow. Doesn't rely on Model Maker or other high-level APIs. This is intended since for python 3.10 and onwards, there are a lot of dependency conflicts ATM. However, it uses tflite_support for metadata writing, which seems unaffected by DC.

The first file you should check is entrenamiento.py (training.py).
Then, loadtflitemodeleinference.
After that, MetaDataWriter.

clasificacion.py is the one that runs on the RPi (Raspberry Pi). You have to upload that code and the .tflite model with the metadata written.
There's also a TomaFotos (TakePhotos), for that same purpose.

All these codes were tested on Python 3.9, and should work in later versions. The rest of the packages are in InformacionRPi (RPiInformation)."
