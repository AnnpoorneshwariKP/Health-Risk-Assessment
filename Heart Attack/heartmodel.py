
import numpy as np 
import pandas as pd 
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import warnings
import seaborn as sns
import pickle
from flask import Flask,request,jsonify,render_template

app=Flask(__name__)
fileh='heart_model'
loaded_model=pickle.load(open(fileh,'rb'))
@app.route('/')
def home():
    return render_template('HeartW.html')
@app.route('/predict',methods=['POST'])
def predict():
    feat=[x for x in request.form.values()]
    fin_feat=[np.array(feat)]
    ot=loaded_model.predict(fin_feat)
    pred=round(ot[0],2)
    if(pred==1):
        return render_template('HeartW.html',prediction="You have a risk of Heart Attack")
    else:
         return render_template('HeartW.html',prediction="You do not have a risk of Heart Attack")
app.run(debug=True)

