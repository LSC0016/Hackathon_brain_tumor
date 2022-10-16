from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from matplotlib import image
from matplotlib.pyplot import imshow
import pandas as pd
import numpy as np
import cv2
import string
from PIL import Image
import keras
import sys, os
from keras.models import load_model
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
import tensorflow

app = Flask(__name__)

def start_model():
    print('initiating model')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
     if request.files['image']:
        img = Image.open(request.files['image'])
        x = np.array(img.resize((128,128)))
        x = x.reshape(1,128,128,3)
        model = load_model('cnn.h5')
        res = model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        if classification == 0:
            names = 'is a tumor'
        else:
            names = 'is not a tumor'
        accuracy = str(res[0][classification]*100) + '% Confidence this ' + names
        return render_template('result.html', title = 'test result', accuracy = accuracy, img = img)


if __name__ == '__main__':
    start_model()
    app.debug = True
    app.run(host='localhost', port=2000)