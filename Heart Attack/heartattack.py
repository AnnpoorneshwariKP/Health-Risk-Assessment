import numpy as np 
import pandas as pd 
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import warnings
import seaborn as sns
import pickle


df = pd.read_csv('heart_reduced.csv')
X = df.drop("output", axis=1)
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=50)


model_dt = DecisionTreeClassifier( max_depth=10, random_state=50)
# Training Model
model_dt.fit(X_train,y_train)

# Making Prediction
pred_dt = model_dt.predict(X_test)
# Calculating Accuracy Score
dt = accuracy_score(y_test, pred_dt)
print(dt)

# Calculating Precision Score
dt = precision_score(y_test, pred_dt)
print(dt)

# Calculating Recall Score
dt = recall_score(y_test, pred_dt)
print(dt)

# Calculating F1 Score
dt = f1_score(y_test, pred_dt)
print(dt)

# confusion Maxtrix
cm2 = confusion_matrix(y_test, pred_dt)
sns.heatmap(cm2/np.sum(cm2), annot = True, fmt=  '0.2%', cmap = 'Reds')

fileh='heart_model'
pickle.dump(model_dt,open(fileh,'wb'))
