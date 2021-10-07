from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
#
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.layers import Embedding
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
# nltk.download ('stopwords')

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


# You can also use pretrained model from Keras
# Check https://keras.io/applications/

# Load arsitektur
model = load_model('modelfix_hoax(lstm).h5', compile=True)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(test_data_padded, model):

    text = request.form['text']
    Berita = ''
    MAX_NB_WORDS=10000
    MAX_SEQUENCE_LENGTH=1000
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    text = text.lower().replace("<br />", " ")
    text=re.sub(strip_special_chars, "", text.lower())
    stop_word_list = set(stopwords.words('english'))
    text = word_tokenize(text)
    new_word_list = []
    for i in text:
        if i not in stop_word_list: new_word_list.append(i)
    #print("Kalimat setelah proses Stopword:", new_word_list)
    clean_test_data = (new_word_list)
    test_tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    test_tokenizer.fit_on_texts(clean_test_data)
    test_sequences = test_tokenizer.texts_to_sequences(clean_test_data)
    word_index = test_tokenizer.word_index
    test_data_padded = pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH, dtype='float32', padding = 'post', value=0.0)
    preds = model.predict(test_data_padded)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('home.html')

@app.route('/identifikasi', methods=['GET', 'POST'])
#def upload():
def sent_anly_prediction():
    if request.method == 'POST':

        text = request.form['text']
        Berita = ''
        MAX_NB_WORDS=1000
        MAX_SEQUENCE_LENGTH=500
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text=re.sub(strip_special_chars, "", text.lower())
        text = word_tokenize(text)
        stop_word_list = set(stopwords.words('english'))
        new_word_list = []
        for i in text:
            if i not in stop_word_list: new_word_list.append(i)
        #print("Kalimat setelah proses Stopword:", new_word_list)

        clean_test_data = (new_word_list)
        test_tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        test_tokenizer.fit_on_texts(clean_test_data)
        test_sequences = test_tokenizer.texts_to_sequences(clean_test_data)
        word_index = test_tokenizer.word_index
        test_data_padded = pad_sequences(test_sequences, dtype='float32', padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
        #preds = model.predict(test_data_padded)
        #return preds

        #print(clean_test_data)
        print('Hasil preprocessing',clean_test_data)
        print(test_data_padded.shape)
        preds = model_predict(test_data_padded,model)
        print(preds[0])

        disease_class = ['HOAX','REAL']
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        probability=str(a[ind]*100)+'%'
        return render_template('home.html', text=clean_test_data, data=result, probability=probability) 
    return None

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
    app.debug=True
    app.run()
