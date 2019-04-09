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
# Create your views here.

def index(request):
    if request.method=='POST':
        image = Images(
            image = request.FILES.get('image')

        )
        image.save()
        images = request.FILES.get('image')
        text = tesseract(images)
        print(text)
        textAnalysis(text)
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

    model = load_model('model.h5')

    y_pred = model.predict(X_test)
    print(y_pred)
    list_t=[]
    for i in range(6):
        if y_pred[i]>=0.5:
            if i==0:
                list_t.append("toxic")
            elif i==1:
                list_t.append("severe toxic")
            elif i==2:
                list_t.append("obscene")
            elif i==3:
                list_t.append("threat")
            elif i==4:
                list_t.append("insult")
            else:
                list_t.append("identity hate")
    ans="The content is "
    for i in range(len(list_t)):
        ans=ans+list_t[i]
        if i!=len(list_t)-1:
            ans=ans+"," 

    print(ans)

    print(y_pred)
    return ans
    # "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"