import numpy as np
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

text = "Manan is obnoxius and should fuck of"

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

print(y_pred.shape)