import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask, render_template,request,flash
import cv2
from werkzeug.utils import secure_filename
import datetime

from cassandra.cluster import Cluster

KEYSPACE = "data"
cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
session = cluster.connect()

try:
    session.execute("create keyspace %s with replication = {'class': 'SimpleStrategy', 'replication_factor': 1};" % KEYSPACE)
except:
    pass
session.set_keyspace('data')
session.execute('use data')
s = session
try:
    s.execute("CREATE TABLE data(uploadfilename text PRIMARY KEY,uploadtime text,prediction text)")
except:
    pass

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    req_time=datetime.datetime.now()
    if request.method == 'POST':
        file = request.files['file']

        upload_filename=secure_filename(file.filename)
        save_filename=upload_filename
        save_filepath=os.path.join(app.root_path,save_filename)
        file.save(save_filepath)

        new_model = keras.models.load_model('C:/Users/luoxiang/saved_model/my_model/my_picmodel.h5')
        probability_model = tf.keras.Sequential([
            new_model,
            tf.keras.layers.Softmax()
        ])
        img=cv2.imread(save_filepath,0)
        size=(28,28)
        img_resize = cv2.resize(img, size)
        img_resize = (np.expand_dims(img_resize,0))
        predictions_single = probability_model.predict(img_resize)
        np.argmax(predictions_single[0])

        params = [save_filename,req_time.strftime('%Y-%m-%d %H:%M:%S.%f'),str(np.argmax(predictions_single[0]))]
        s.execute("INSERT INTO data (uploadfilename,uploadtime,prediction)VALUES (%s, %s,%s)", params)

        return("%s%s%s%s%s%s%s%s%s"%("Upload File Name:",save_filename,"\n","Upload Time:",req_time,"\n","Prediction:",np.argmax(predictions_single[0]),"\n"))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

