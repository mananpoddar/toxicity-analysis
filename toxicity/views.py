from django.shortcuts import render
from .models import Images
import cv2
import sys
import pytesseract
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import tensorflow as tf
import os
import pyttsx3;
model = load_model('model.h5')
graph = tf.get_default_graph()
# Create your views here.

def index(request):
    if request.method=='POST':
        image = Images(
            image = request.FILES.get('image')

        )
        image.save()
        images = request.FILES.get('image')
        input_text = request.POST.get('text')
        text =" "
        final_result=" "
        print(input_text)
        print(images)
        if input_text!=None:
            final_result = textAnalysis(input_text)
            text = input_text
        elif images!=" ":
            text = tesseract(images)
            final_result = textAnalysis(text)
        # engine = pyttsx3.init();
        # engine.say(final_result);
        # engine.runAndWait() ;
        results=final_result.split(' ')
        for result in results:
            os.system("espeak "+result)
        arr = final_result.split('-')
        final_result = " ".join(arr)
        return render(request,"toxicity/result.html",{"final_result":final_result,"text":text})
    return render(request,"toxicity/index.html")


def tesseract(images):
    config = ('-l eng --oem 1 --psm 3')
    path = "./media/images/"+str(images)
    images = path
    im = cv2.imread(images,cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(im,config=config)
    return  text


def textAnalysis(text):
    max_features = 20000
    maxlen = 100
    tokenizer = Tokenizer(num_words=max_features)

    model_file = "object.sav"
    with open(model_file,mode='rb') as model_f:
        tokenizer = pickle.load(model_f)

    l=[]
    l.append(text)
    text = np.array(l)

    # arr = np.array()
    # arr.append(text)
    # text = arr

    tokenized_test = tokenizer.texts_to_sequences(text)

    X_test = pad_sequences(tokenized_test, maxlen=maxlen)
    global graph,model
    with graph.as_default():

        y_pred = model.predict(X_test)
        print(y_pred)
        list_t=[]
        for i in range(6):
            if y_pred[0][i]>=0.5:
                if i==0:
                    list_t.append(" toxic")
                elif i==1:
                    list_t.append(" severe-toxic")
                elif i==2:
                    list_t.append(" obscene")
                elif i==3:
                    list_t.append(" threat")
                elif i==4:
                    list_t.append(" insult")
                else:
                    list_t.append(" identity-hate")
        ans="The-content-is"
        for i in range(len(list_t)):
            ans=ans+list_t[i]
            if i!=len(list_t)-1:
                ans=ans+","
        if(len(list_t)==0):
            ans = ans + "not-at-all-toxic or-lies-under-any-of-the-toxic-categories"

        return ans
        # "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
