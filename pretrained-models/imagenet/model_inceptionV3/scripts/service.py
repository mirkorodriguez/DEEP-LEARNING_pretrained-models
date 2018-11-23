#Import Flask
from flask import Flask, request
#Import Keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#inception_v3
from keras.applications import inception_v3
from keras.applications.imagenet_utils import decode_predictions
#Import python files
from inceptionV3_loader import cargarModeloInceptionV3
import numpy as np

#Initialize the application service
app = Flask(__name__)
global inceptionV3_model, inceptionV3_graph
inceptionV3_model, inceptionV3_graph = cargarModeloInceptionV3()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la URP - RNA!  Inception V3 Model'

@app.route('/inceptionv3/predict/',methods=['GET','POST'])
def default():
	#Default image
	filename = '../../samples/perro.jpg'
	# load an image in PIL format
	original = load_img(filename, target_size=(229, 229))
	print('PIL image size',original.size)

	numpy_image = img_to_array(original)
	print('numpy array size',numpy_image.shape)

	image_batch = np.expand_dims(numpy_image, axis=0)
	print('image batch size', image_batch.shape)

	#=======================
	#Prediccion de Imagenes
	#=======================
	with inceptionv3_graph.as_default():
		# prepare the image for the Inception model
		processed_image = inception_v3.preprocess_input(image_batch.copy())
		# get the predicted probabilities for each class
		predictions = inceptionV3_model.predict(processed_image)
		# convert the probabilities to class labels
		label_inceptionV3 = decode_predictions(predictions)
		print(label_inceptionV3)
		return label_inceptionV3

# Run de application
app.run(host='0.0.0.0',port=5000)
