import numpy as np 
import pandas as pd 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pickle
from flask import Flask,request,jsonify,render_template

app=Flask(__name__)
fileo='obesity_model'
loaded_omodel=pickle.load(open(fileo,'rb'))
@app.route('/')
def home():
    return render_template('ObesityW.html')
@app.route('/predict',methods=['POST'])
def predict():
    feat=[x for x in request.form.values()]
    fin_feat=[np.array(feat)]
    ot=loaded_omodel.predict(fin_feat)
    pred=round(ot[0],7)
    if(pred==0):
        return render_template('ObesityW.html',prediction="You are under weight")
    elif(pred==1):
         return render_template('ObesityW.html',prediction="You have normal weight")
    elif(pred==2):
        return render_template('ObesityW.html',prediction="You have Overweight Level-1")
    elif(pred==3):
        return render_template('ObesityW.html',prediction="You have Overweight Level-2")
    elif(pred==4):
        return render_template('ObesityW.html',prediction="You have Obesity Type-1")
    elif(pred==5):
        return render_template('ObesityW.html',prediction="You have Obesity Type-2")
    else:
        return render_template('ObesityW.html',prediction="You have Obesity Type-3")
app.run(debug=True)

 
