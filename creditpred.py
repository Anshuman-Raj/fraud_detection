# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:28:12 2021

@author: Anshuman Raj
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import imblearn as imb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb

#settings for xgboost model
param = {
    'max_depth': 8,
    'eta': 0.03,
    'objective': 'binary:hinge',
    'num_class': 1} 
epochs = 25


#opening and reading the data file
data_org=pd.read_csv('creditcard.csv')
data_copy=data_org.copy()
data_copy=data_copy.sample(frac=1).reset_index(drop=True)
df=data_copy.to_numpy()
x,y=df[:,:30],df[:,30]
print('Original number  of samples: '+str(len(y)))


#oversampling with smote technique to remove the imbalance
oversample=imb.over_sampling.SMOTE()
x, y = oversample.fit_resample(x, y)
print('Number  of samples after SMOTE: '+str(len(y)))


#preparing data to pass into the model
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

#training and predecting with xgboost modedl
model = xgb.train(param, train, epochs)
predictions = model.predict(test)

#Dictinoary to convert numerical category to readable format
dict={1:"Scam/Fraud", 0:"Fair Transaction"}

#Displaying the Predictionns
while True:
    expec=np.random.randint(len(predictions))
    print('\n\nAmount: $%s' % X_test[expec][-1])
    print('Type of transaction: ' + dict[int(y_test[expec])])
    print('Predicted type: '+dict[int(predictions[expec])])
    ans = input("\nContinue? [Y/n]")
    if ans and ans.lower().startswith('n'):
      break
    
    
print('Acuuracy of prediction = %s' % (accuracy_score(y_test, predictions)*100), end="%\n")