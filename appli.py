# -*- coding: utf-8 -*-
#from flask import Flask, request, jsonify
import Flask
import request
import jsonify
import pandas as pd
import numpy as np
import joblib
import json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
app.config["DEBUG"] = True


path = "C:\\Users\\toure\\Desktop\\OpenClassrooms\\Projet 7\\donnees\\"

loaded_model = joblib.load('C://Users//toure//Desktop//OpenClassrooms//Projet 7//logreg_housing.joblib')

#************test*********************
num_rows = 10000
data = pd.read_csv(path+"donnees_traites.csv", nrows= num_rows)
data_client_1 = pd.read_csv(path+"donnees_traites_1.csv")
colonnes_fr = pd.read_csv(path+"colonnes.csv")

data = data[list(colonnes_fr.iloc[:,1])]
data_client_1 = data_client_1[list(colonnes_fr.iloc[:,1])]

data_client = pd.concat([data, data_client_1])
data_client.reset_index(drop=True, inplace=True)

data_client = data_client.drop(['TARGET'], axis=1)
references_test = data_client.SK_ID_CURR
#*************************************


@app.route('/')
def welcome():
    return "Welcome all"

@app.route('/predict')
def predict_credit():
    ref_client=request.args.get('SK_ID_CURR')
    
    rt = list(references_test).index(int(ref_client))
    client = data_client[rt:rt+1]
    prediction=loaded_model.predict_proba(client)[0][1]
    
    return str(prediction)


    
if __name__=='__main__':
        app.run()