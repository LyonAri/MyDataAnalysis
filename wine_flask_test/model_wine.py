import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import os

print(os.getcwd())
path="C:\\Users\\user\\ict_class\\kaggle4th_flask_ml\\wine_flask_test\\"
df = pd.read_csv(path + "data\wine.csv")
print(df.shape)
print(df.columns)

print(df.isnull().sum())        #결측치 없음

sel = ['pH', 'chlorides', 'fixed acidity', 'volatile acidity', 'citric acid', 'density']

X = df[sel]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.3, 
                                                random_state=30)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print("-----------")

model_gd = GradientBoostingClassifier().fit(X_train, y_train)
gd_pred = model_gd.predict(X_test)
gd_acc = model_gd.score(X_test, y_test)
print("train(gd) : ", model_gd.score(X_train, y_train))
print("test(gd) : ", gd_acc)

gd = GradientBoostingClassifier().fit(X_train, y_train)
score = cross_val_score(gd, X_test, y_test, cv=2, scoring='accuracy')
print('교차 검증 점수 : ', np.mean(score))

print("-----------")

model_rf = RandomForestClassifier().fit(X_train, y_train)
rf_pred = model_rf.predict(X_test)
rf_acc = model_rf.score(X_test, y_test)
print("train(rf) : ", model_rf.score(X_train, y_train))
print("test(rf) : ", rf_acc)

rf = RandomForestClassifier().fit(X_train, y_train)
score = cross_val_score(rf, X_test, y_test, cv=2, scoring='accuracy')
print('교차 검증 점수 : ', np.mean(score))

pickle.dump(rf, open(path+'\\wine_model\\wine_base.pkl', 'wb'))