import flask
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
import os
from tensorflow.keras.applications.vgg16 import VGG16
import pandas as pd
import csv
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
from flask_cors import CORS, cross_origin

from flask import Flask,request
import time
import json
import os
import base64
import shutil
from sklearn import metrics
from sklearn.metrics import confusion_matrix

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

global logstr
logstr=''
@app.route('/')
@cross_origin()
def hello_world():  # put application's code here
    return {'data':'Hello World!'}

@app.route('/getlogs')
@cross_origin()
def getlogs():  # put application's code here
    return json.dumps(logstr)

@app.route('/test',methods=['POST'])
@cross_origin()
def test():  # put application's code here
    data=(json.loads(request.data))["data"]
    count=0;
    if os.path.exists("Test")==True:
        shutil.rmtree("Test")
    for d in data:
        os.makedirs("Test/"+d["label"], exist_ok=True)
        with open("Test/"+d["label"]+"/"+d["name"]+".png", "wb") as fh:
            print(bytes(d["data"].split(',')[1],'utf-8'))
            fh.write(base64.decodebytes(bytes(d["data"].split(',')[1],'utf-8')))
    SIZE = 256  # Resize images
    test_images = []

    test_labels = []
    test_image_names=[]
    for directory_path in glob.glob("Test/*"):
        label = directory_path.split("/")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            test_images.append(img)
            test_image_names.append(img_path.split("\\")[-1])
            test_labels.append(label)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    le = preprocessing.LabelEncoder()
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)
    x_test, y_test =  test_images, test_labels_encoded
    x_test= x_test / 255.0
    y_test_one_hot = to_categorical(y_test)

    test_feature_extractor = VGG_model.predict(x_test)
    test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)

    test_PCA = pca.transform(test_features)

    predict_test = model.predict(test_PCA)
    predict_test = np.argmax(predict_test, axis=1)
    predict_test = le.inverse_transform(predict_test)

    print("Accuracy = ", metrics.accuracy_score(test_labels, predict_test))
    print("Confusion Matrix = ", confusion_matrix(test_labels, predict_test))
    print("Precision = ", metrics.precision_score(test_labels, predict_test, average='micro'))
    print("Recall = ", metrics.recall_score(test_labels, predict_test, average='micro'))
    print("classification report",metrics.classification_report(test_labels, predict_test, digits=3))
    accuracy= str(metrics.accuracy_score(test_labels, predict_test))
    cm= str(confusion_matrix(test_labels, predict_test))
    precision= str(metrics.precision_score(test_labels, predict_test, average='macro'))
    recall= str(metrics.recall_score(test_labels, predict_test, average='macro'))
    f1score= str(metrics.f1_score(test_labels, predict_test, average='macro'))
    print(test_labels)
    return '{"accuracy":'+accuracy+',"precision":'+precision+',"f1score":'+f1score+',"recall":'+recall+',"testfilenames":'+str(test_image_names).replace("'","\"")+',"testlabels":'+str(test_labels).replace("'","\"").replace(" ",",")+',"predictedlabels":'+str(predict_test).replace(" ",",").replace("'","\"")+',"cm":'+cm.replace("[ ","[").replace("  ",",").replace(" ",",")+'}'

@app.route('/validate',methods=['POST'])
@cross_origin()
def validate():  # put application's code here
    data=(json.loads(request.data))["data"]
    count=0;
    if os.path.exists("Validate")==True:
        shutil.rmtree("Validate")
    for d in data:
        os.makedirs("Validate/"+d["label"], exist_ok=True)
        with open("Validate/"+d["label"]+"/"+d["name"]+".png", "wb") as fh:
            print(bytes(d["data"].split(',')[1],'utf-8'))
            fh.write(base64.decodebytes(bytes(d["data"].split(',')[1],'utf-8')))
    SIZE = 256  # Resize images
    val_images = []

    val_labels = []
    val_image_names=[]
    for directory_path in glob.glob("Validate/*"):
        label = directory_path.split("/")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            val_images.append(img)
            val_image_names.append(img_path.split("\\")[-1])
            val_labels.append(label)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    le = preprocessing.LabelEncoder()
    le.fit(val_labels)
    val_labels_encoded = le.transform(val_labels)
    x_val, y_val =  val_images, val_labels_encoded
    x_val= x_val / 255.0
    y_val_one_hot = to_categorical(y_val)

    val_feature_extractor = VGG_model.predict(x_val)
    val_features = val_feature_extractor.reshape(val_feature_extractor.shape[0], -1)

    val_PCA = pca.transform(val_features)

    predict_val = model.predict(val_PCA)
    predict_val = np.argmax(predict_val, axis=1)
    predict_val = le.inverse_transform(predict_val)

    print("Accuracy = ", metrics.accuracy_score(val_labels, predict_val))
    print("Confusion Matrix = ", confusion_matrix(val_labels, predict_val))
    print("Precision = ", metrics.precision_score(val_labels, predict_val, average='micro'))
    print("Recall = ", metrics.recall_score(val_labels, predict_val, average='micro'))

    accuracy= str(metrics.accuracy_score(val_labels, predict_val))
    cm= str(confusion_matrix(val_labels, predict_val))
    precision= str(metrics.precision_score(val_labels, predict_val, average='micro'))
    recall= str(metrics.recall_score(val_labels, predict_val, average='micro'))
    f1score= str(metrics.f1_score(val_labels, predict_val, average='micro'))
    print(val_labels)
    return '{"accuracy":'+accuracy+',"precision":'+precision+',"f1score":'+f1score+',"recall":'+recall+',"valfilenames":'+str(val_image_names).replace("'","\"")+',"vallabels":'+str(val_labels).replace("'","\"").replace(" ",",")+',"predictedlabels":'+str(predict_val).replace(" ",",").replace("'","\"")+',"cm":'+cm.replace("[ ","[").replace("  ",",").replace(" ",",")+'}'



@app.route('/train',methods=['POST'])
@cross_origin()
def train():  # put application's code here
    data=(json.loads(request.data))["data"]
    count=0;
    shutil.rmtree("Train")
    for d in data:
        os.makedirs("Train/"+d["label"], exist_ok=True)
        count=count+1
        with open("Train/"+d["label"]+"/image"+str(count)+".png", "wb") as fh:
            print(bytes(d["data"].split(',')[1],'utf-8'))
            fh.write(base64.decodebytes(bytes(d["data"].split(',')[1],'utf-8')))
    SIZE = 256  # Resize images
    train_images = []

    train_labels = []
    for directory_path in glob.glob("Train/*"):
        label = directory_path.split("/")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
            train_labels.append(label)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    le = preprocessing.LabelEncoder()
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)
    x_train, y_train = train_images, train_labels_encoded
    x_train= x_train / 255.0
    y_train_one_hot = to_categorical(y_train)
    global VGG_model
    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
    for layer in VGG_model.layers:
        layer.trainable = False
    VGG_model.summary()
    train_feature_extractor = VGG_model.predict(x_train)
    train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0], -1)
    n_PCA_components = len(x_train)
    global pca
    pca = PCA(n_components=n_PCA_components)
    train_PCA = pca.fit_transform(train_features)
    global model
    model = Sequential()
    inputs = Input(shape=(n_PCA_components,))  # Shape = n_components
    hidden = Dense(256, activation='relu')(inputs)
    output = Dense(len(np.unique(train_labels)), activation='softmax')(hidden)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    logs=model.fit(train_PCA, y_train_one_hot, epochs=100, verbose=2)
    global logstr
    #for e in range(len(train_PCA)):
    #    logs=model.fit(train_PCA, y_train_one_hot, epochs=1, batch_size=len(train_PCA), verbose=2)
    #    time.sleep(2)
    logstr=str(logs.history)

    return json.dumps(logstr)

if __name__ == '__main__':
    app.run()
