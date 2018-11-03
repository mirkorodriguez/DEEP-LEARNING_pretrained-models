#Import Flask
from flask import Flask
from keras.preprocessing import image
from cnn_executor import cargarModelo
import numpy as np

#Initialize the application service
app = Flask(__name__)
global loaded_model, graph
loaded_model, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la URP - RNA!'

@app.route('/rayos-x/', methods=['GET','POST'])
def rayosx():
	return 'Modelo Rayos-X!'

@app.route('/rayos-x/default/', methods=['GET','POST'])
def rayosxy():
	# dimensions of our images.
	img_width, img_height = 299, 299
	# Show
	img_path='../samples/img1.png'
	img = image.load_img(img_path, target_size=(img_width, img_height))
	img = image.img_to_array(img)
	x = np.expand_dims(img, axis=0) * 1./255

	with graph.as_default():
		score = loaded_model.predict(x)
		if score < 0.5
			resultado = 'Prediccion: Abdomen X-ray , score: ' + score
		else
		    resultado = 'Prediccion: Pulmon X-ray , score: ' + score
		print('Prediccion:', score, 'Abdomen X-ray' if score < 0.5 else 'Pulmon X-ray')
		return resultado

# Run de application
app.run(host='0.0.0.0',port=5000)
