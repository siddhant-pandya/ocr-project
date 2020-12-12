from django.shortcuts import render
from django.http import HttpResponse
import csv
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.templatetags.static import static
from django.contrib.staticfiles.finders import find


import os
import fnmatch
import cv2
import numpy as np
import string
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import model_from_json


def func(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    #img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)   
    w, h = img.shape
    if h > 128 or w > 32:
        c = 0
    else:
        if w < 32:
            add_zeros = np.ones((32-w, h))*255
            img = np.concatenate((img, add_zeros))
        if h < 128:
            add_zeros = np.ones((32, 128-h))*255
            img = np.concatenate((img, add_zeros), axis=1)
    img = np.expand_dims(img , axis = 2)
    img = img/255.
    img = np.array(img)
    predict_img  = []
    predict_img.append(img)
    predict_img = np.array(predict_img)
    
    json_file = open('static/ocrapp/model.json', 'r')
    
    loaded_model_json = json_file.read()
    #print(loaded_model_json)
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("static/ocrapp/best_model.hdf5")
    
    prediction = loaded_model.predict(predict_img[:1])
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])
    char_list = string.ascii_letters+string.digits
    i = 0
    s = ""
    for x in out:
        for ps in x:  
            if int(ps) != -1:
                s +=(char_list[int(ps)])  
        i+=1
    return s

# Create your views here.
def index(request):
    s=""
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        #fs = FileSystemStorage()
        #filename = fs.save(myfile.name, myfile)
        #uploaded_file_url = fs.url(filename)
        img = cv2.imdecode(np.fromstring(request.FILES['myfile'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        s = func(img)
        

        # return render(request, 'ocrapp/index.html', {
        #     'uploaded_file_url': uploaded_file_url, 'txt': arranged_text
        # })

    context = {'txt':s}
    return render(request, 'ocrapp/index.html', context)
