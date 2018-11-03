#Import Flask
from flask import Flask

#Initialize the application service
app = Flask(__name__)

#Define a route
@app.route('/rayos-x/')
def hello_world():
	return 'Hello World!'

# Run de application
app.run(host='0.0.0.0',port=5000)
