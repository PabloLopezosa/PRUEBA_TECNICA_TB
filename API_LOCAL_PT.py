from regex import F
from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import pickle
import os
import re
from FUNCIONES import signs_tweets, remove_stopwords, spanish_stemmer


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

model = pickle.load(open('Data/finished_model.model','rb'))

@app.route('/', methods=['GET'])
def home():
    return "MODELO MACHINE LEARNING ANALISIS SENTIMIENTOS TWITTER"


@app.route('/api/v1/consulta', methods=['GET'])
def consulta():

    tweet = request.args.get('tweet', None)

    #signs_tweets(tweet)
    #remove_stopwords(tweet)
    #spanish_stemmer(tweet)

    predictions = model.predict([tweet])[0]

    if predictions == 1:
        return  'Es un tweet negativo'
    elif predictions == 0:
        return 'Es un tweet positivo'
 
#http://127.0.0.1:5000/api/v1/consulta?tweet=genial%20en%20thebridge
#http://127.0.0.1:5000/api/v1/consulta?tweet=fatal%20en%20thebridge

#https://lopezosa97.pythonanywhere.com/api/v1/consulta?tweet=genial%20la%20experiencia%20en%20the%20bridge
#https://lopezosa97.pythonanywhere.com/api/v1/consulta?tweet=fatal%20la%20experiencia%20en%20the%20bridge

app.run()