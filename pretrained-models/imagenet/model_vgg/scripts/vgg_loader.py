# Credits:
# -----------------------------------------------------------
# Cargando modelo de disco
import tensorflow as tf
from keras.applications import vgg16

def cargarModeloVGG():
    #Load the VGG model
    print("Cargando modelo VGG ...")
    vgg_model = vgg16.VGG16(weights='imagenet')
    vgg_model.summary()
    print("Modelo VGG cargado!")
    vgg_graph = tf.get_default_graph()
    return vgg_model, vgg_graph
