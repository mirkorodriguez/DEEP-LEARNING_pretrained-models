#Import Flask
from flask import Flask, request, jsonify, redirect
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

import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../../samples/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#Initialize the application service
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global inceptionV3_model, inceptionV3_graph
inceptionV3_model, inceptionV3_graph = cargarModeloInceptionV3()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la URP - RNA!  Inception V3 Model'

@app.route('/inceptionv3/default/',methods=['GET','POST'])
def default():

	data = {"success": False}

	#Default image
	filename = '../../samples/image01.jpg'
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
	with inceptionV3_graph.as_default():
		# prepare the image for the Inception model
		processed_image = inception_v3.preprocess_input(image_batch.copy())
		# get the predicted probabilities for each class
		predictions = inceptionV3_model.predict(processed_image)
		# convert the probabilities to class labels
		label_inceptionV3 = decode_predictions(predictions)
		print(label_inceptionV3)

		#Results as Json
		data["predictions"] = []
		for (imagenetID, label, prob) in label_inceptionV3[0]:
			r = {"label": label, "probability": float(prob)}
			data["predictions"].append(r)

		#Success
		data["success"] = True

	return jsonify(data)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/inceptionv3/predict/',methods=['POST'])
def predict():

	data = {"success": False}

	if request.method == "POST":
		# check if the post request has the file part
		if 'file' not in request.files:
			print('No file part')
		file = request.files['file']
		# if user does not select file, browser also submit a empty part without filename
		if file.filename == '':
			print('No selected file')
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

			###############################################################
			#loading image
			filename = UPLOAD_FOLDER + '/' + filename

			original = load_img(filename, target_size=(229, 229))
			print('PIL image size',original.size)

			numpy_image = img_to_array(original)
			print('numpy array size',numpy_image.shape)

			image_batch = np.expand_dims(numpy_image, axis=0)
			print('image batch size', image_batch.shape)

			#=======================
			#Prediccion de Imagenes
			#=======================
			with inceptionV3_graph.as_default():
				# prepare the image for the Inception model
				processed_image = inception_v3.preprocess_input(image_batch.copy())
				# get the predicted probabilities for each class
				predictions = inceptionV3_model.predict(processed_image)
				# convert the probabilities to class labels
				label_inceptionV3 = decode_predictions(predictions)
				print(label_inceptionV3)
			###############################################################

				#Results as Json
				data["predictions"] = []
				for (imagenetID, label, prob) in label_inceptionV3[0]:
					r = {"label": label, "probability": float(prob)}
					data["predictions"].append(r)

				#Success
				data["success"] = True

	return jsonify(data)

# Run de application
app.run(host='0.0.0.0',port=5000)
