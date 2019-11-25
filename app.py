from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPool2D, ReLU
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
from keras.models import Sequential
import cv2
import os
from os import listdir, remove
import math
import re

app = Flask(__name__)

model = Sequential()
graph = tf.get_default_graph()
    
@app.route('/images', methods = ['POST'])
def upload_file():
    save_images(request)
    X_pred = get_test_images('./')
    remove_images('./')
    predictions = get_predictions(np.array([first for first, _ in X_pred]))
    items = {}
    for i, _ in enumerate(X_pred):
      items[X_pred[i][1]] = predictions[i].__str__()
    print(items)
    res = create_response(items)
    return jsonify(res)

		
def save_images(request):
  f = request.files.getlist('images')
  for i, file in enumerate(f):
    file.save(file.filename + ".png")

def remove_images(dir_name):
  files_names = [file for file in listdir(dir_name)]
  for file_name in files_names:
    if re.search('([-\w]+\.(?:jpg|gif|png|jpeg|JPG))', file_name):
      remove(file_name)

def get_test_images(dir_name):
  files_names = [file for file in listdir(dir_name)]
  X = [(file_name, cv2.resize(cv2.imread(file_name, cv2.IMREAD_GRAYSCALE), (32, 50))) for file_name in files_names if re.search('([-\w]+\.(?:jpg|gif|png|jpeg|JPG))', file_name)]
  X = [(np.expand_dims(second, axis=3), first) for first, second in X]
  return X


def create_response(items):
  list = []
  for key, value in items.items():
    obj = {
      'faceId': key[0: key.index('.png')],
      'emotion': value
    }
    list.append(obj)
  return {'results': list}

def get_predictions(X_pred):
    global model
    global graph
    with graph.as_default():
        return np.argmax(model.predict(X_pred), axis=1)


def prepare_keras_model():
    global model
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(50, 32, 1), kernel_regularizer=l2(0.01)))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2, 2)))    
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.load_weights('my_checkpoint')


if __name__ == "__main__":
    prepare_keras_model()
    app.run(debug = True)

