import numpy as np 
import pandas as pd 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pickle

df = pd.read_csv('ObesityData.csv')
X = df.drop("NObeyesdad", axis=1)
y = df['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=50)

seed = 50
num_trees =60


#Creating Model Object
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)

#Training the model
model.fit(X_train,y_train)

#Making Prediction
pred_dt = model.predict(X_test)

# Calculating Accuracy Score
dt = accuracy_score(y_test, pred_dt)
print(dt)

# Calculating Precision Score
dt = precision_score(y_test, pred_dt,average='micro')
print(dt)

# Calculating Recall Score
dt = recall_score(y_test, pred_dt,average='macro')
print(dt)

# Calculating F1 Score
dt = f1_score(y_test, pred_dt,average='weighted')
print(dt)

# confusion Maxtrix
cm2 = confusion_matrix(y_test, pred_dt)
sns.heatmap(cm2/np.sum(cm2), annot = True, fmt=  '0.2%', cmap = 'Reds')

fileo='obesity_model'
pickle.dump(model,open(fileo,'wb'))



 
