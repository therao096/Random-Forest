# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:35:45 2020

@author: Varun
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\RANDOM FORESTS\\Company_Data.csv")
data.Sales=pd.cut(data.Sales,bins=[0,4,8,12],labels=['A','B','C'])
colnames=list(data.columns)
le=LabelEncoder()

data['ShelveLoc']=le.fit_transform(data['ShelveLoc'])
data['Urban']=le.fit_transform(data['Urban'])
data['US']=le.fit_transform(data['US'])
data=data.dropna(subset=['Sales'])
predictors= colnames[1:]
target= colnames[0]
train,test= train_test_split(data,test_size=0.3,random_state=0)
X=train[predictors]
Y=train[target]

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion="entropy")
rf.fit(X,Y)
rf.estimators_
rf.classes_
rf.n_classes_ ##3
rf.n_features_ ##10
rf.n_outputs_  ##1
rf.oob_score

rf.predict(X)
train['rf_predict']= rf.predict(X)
train['Sales']
cols=['Sales','rf_predict']

from sklearn.metrics import confusion_matrix
confusion_matrix(train['Sales'],train['rf_predict'])
pd.crosstab(train['Sales'],train['rf_predict'])
####accuracy of training model is 100%


####checking for test data
TEST_X=test[predictors]
TEST_Y=test[target]
rf.fit(X,Y)
rf.predict(TEST_X)
test['rf_predict']=rf.predict(TEST_X)
test['Sales']
cols_test=['Sales','rf_predict']
confusion_matrix(test['Sales'],test['rf_predict']) 
(50+26)/(11+1+50+8+16+26)
####67.85% accuracy