# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 19:22:21 2020

@author: Varun
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data= pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\RANDOM FORESTS\\Fraud_Check.csv")
le=LabelEncoder()
data['Undergrad']=le.fit_transform(data['Undergrad'])
data['Marital.Status']=le.fit_transform(data['Marital.Status'])
data['Urban']=le.fit_transform(data['Urban'])
data.rename(columns={'Taxable.Income':'TaxableIncome'},inplace=True)
data['TaxableIncome'].max()
data['TaxableIncome']=pd.cut(data.TaxableIncome, bins=[0,30000,99619], labels=['Risky','Good'])
data=data[['Undergrad','Marital.Status','City.Population','Work.Experience','Urban','TaxableIncome']]
data.shape
data['TaxableIncome']=le.fit_transform(data['TaxableIncome'])
colnames=list(data.columns)
predictors=colnames[0:5]
target=colnames[5]
train,test= train_test_split(data,test_size=0.3,random_state=0)
X=train[predictors]
Y=train[target]
rf= RandomForestClassifier(n_estimators=100,oob_score=True,n_jobs=3,criterion="entropy")
rf.fit(X,Y)
rf.predict(X)
train['rf_predict']=rf.predict(X)
pd.crosstab(train['TaxableIncome'],train['rf_predict']) ###100%

###testingdata
test_x=test[predictors]
test_y=test[target]
rf.fit(X,Y)
test['rf_predict']=rf.predict(test_x)
pd.crosstab(test['TaxableIncome'], test['rf_predict'])
(129+3)/(129+3+8+40)
####73.333% of accuracy