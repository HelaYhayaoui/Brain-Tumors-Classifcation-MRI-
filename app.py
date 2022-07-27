from flask import Flask, render_template, request

# example of loading an image with the Keras API
from keras.preprocessing import image
# example of loading an image with the Keras API

from tensorflow.keras.utils import load_img

import numpy as np
import keras
import cv2
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

app =   Flask(__name__)

def get_model():
    global model
    model = keras.models.load_model('MYmodel.h5')
    print("Model loaded!")

classes={0: ' Tumor Type: Glioma Tumor', 1: 'Tumor Type: Meningioma Tumor', 2: 'It is Not a Tumor!', 3: 'Tumor Type: Pituitary Tumor'}



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print('hello model')
    get_model()
    if request.method == 'POST':
        file = request.files['file']
        # The filename attribute in the FileStorage provides the filename submitted by the client.
        img_path = os.path.join('static/images', secure_filename(file.filename))
        file.save(img_path)
        img = load_img(img_path, target_size=(120, 120))
        img = np.array(img)
        img = img.reshape(1, 120,120,3)
        p = model.predict(img)
        output = np.argmax(p, axis=1)
        return render_template('result.html', prediction_text=classes[output[0]], img_path = img_path)



if __name__ == "__main__":
    app.run(debug=True)

