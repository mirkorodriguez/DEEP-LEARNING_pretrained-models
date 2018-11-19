# Credits:
# -----------------------------------------------------------
# Cargando modelo de disco
import tensorflow as tf
from keras.applications import vgg16

def cargarModelo():
    #Load the VGG model
    print("Cargando modelo ...")
    vgg_model = vgg16.VGG16(weights='imagenet')
    print("Modelo cargado!")
    vgg_graph = tf.get_default_graph()
    return vgg_model, vgg_graph
