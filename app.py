import flask
from flask import Flask,render_template,url_for,request
import pickle
import base64
import numpy as np
import cv2


#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

#Our dictionary
label_dict = {0:'Cat', 1:'Giraffe', 2:'Sheep', 3:'Bat', 4:'Octopus', 5:'Camel'}

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__)

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')

#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
        final_pred = 'lol'
        return render_template('results.html', prediction =final_pred)

if __name__ == '__main__':
	app.run(debug=True)