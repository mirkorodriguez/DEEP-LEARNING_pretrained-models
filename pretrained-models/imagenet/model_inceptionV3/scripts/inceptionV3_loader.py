# Credits:
# -----------------------------------------------------------
# Cargando modelo de disco
import tensorflow as tf
from keras.applications import inception_v3

def cargarModeloInceptionV3():
    #Load the InceptionV3 model
    print("Cargando modelo InceptionV3 ...")
    inceptionV3_model = inception_v3.InceptionV3(weights='imagenet')
    inceptionV3_model.summary()
    print("Modelo InceptionV3 cargado!")
    inceptionV3_graph = tf.get_default_graph()
    return inceptionV3_model, inceptionV3_graph
