from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask, request

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        pic = request.files['file']
        classify(pic)
    return None

def classify(pic):
    new_model = keras.models.load_model('C:/Users/luoxiang/saved_model/my_model')
    probability_model = tf.keras.Sequential([
        new_model,
        tf.keras.layers.Softmax()
    ])
    image = tf.image.decode_jpeg(pic, channels = 1)
    image = tf.image.resize(image, [28, 28])
    im = np.array(image, dtype=int)
    im = im.reshape(28, 28)
    im = np.array([im])
    predictions1 = probability_model.predict(im)
    return np.argmax(predictions1[0])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

