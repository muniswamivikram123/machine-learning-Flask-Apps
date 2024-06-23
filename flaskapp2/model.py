# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:14:14 2024

@author: Vikra
"""

import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv('heart_data.csv')
print(df.head())

df = df.drop("Unnamed: 0",axis=1)
sns.lmplot(x='biking',y='heart.disease',data=df)
sns.lmplot(x='smoking',y='heart.disease',data=df)

x_df = df.drop('heart.disease',axis=1)
y_df = df['heart.disease']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x_df,y_df,test_size = 0.3,random_state = 42)
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
print(model.score(X_train,y_train))
prediction_test = model.predict(X_test)
print(y_test,prediction_test)
print("mean sq error between y_test and predicted = ",np.mean(prediction_test - y_test)**2)

import pickle
pickle.dump(model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[70.1,26.3]]))