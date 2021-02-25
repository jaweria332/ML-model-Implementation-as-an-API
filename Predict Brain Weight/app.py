# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 02:53:20 2021

@author: dell
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
pickle_in = open("classifier.pkl", 'rb')
classifier=pickle.load(pickle_in)




@app.route('/')
def welcome():
    return "Welcome to Flask Learning for Docker"



@app.route('/predict_head')
def predict_brain_weight():
    headsize = request.args.get('Head Size(cm^3)')
    prediction = classifier.predict([headsize])
    return "Predictions (by model) = " + str(prediction)



@app.route('/predict_head_file', methods=["POST"])
def predict_brain_weight_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "Predictions (by model) = " + str(list(prediction))

if __name__=='__main__':
    app.run()