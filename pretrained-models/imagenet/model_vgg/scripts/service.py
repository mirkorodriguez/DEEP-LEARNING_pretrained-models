#Import Flask
from flask import Flask, request
#Import Keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#VGG
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
#Import python files
from vgg_loader import cargarModeloVGG
import numpy as np

#Initialize the application service
app = Flask(__name__)
global vgg_model, vgg_graph
vgg_model, vgg_graph = cargarModeloVGG()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la URP - RNA!  VGG Model'

@app.route('/vgg/predict/',methods=['GET','POST'])
def default():
	#Default image
	filename = '../../samples/image01.jpg'
	# load an image in PIL format
	original = load_img(filename, target_size=(224, 224))
	print('PIL image size',original.size)

	numpy_image = img_to_array(original)
	print('numpy array size',numpy_image.shape)

	image_batch = np.expand_dims(numpy_image, axis=0)
	print('image batch size', image_batch.shape)

	#=======================
	#Prediccion de Imagenes
	#=======================
	with vgg_graph.as_default():
		#VGG16 Network
		# prepare the image for the VGG model
		processed_image = vgg16.preprocess_input(image_batch.copy())
		# get the predicted probabilities for each class
		predictions = vgg_model.predict(processed_image)
		print (predictions)
		# convert the probabilities to class labels
		# We will get top 5 predictions which is the default
		label_vgg = decode_predictions(predictions)
		print (label_vgg)
		return label_vgg

# Run de application
app.run(host='0.0.0.0',port=5000)
