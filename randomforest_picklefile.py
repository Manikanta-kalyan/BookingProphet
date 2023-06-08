#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

dataset_clean = pd.read_csv(absolute+path + '/complete_clean.csv')
print(dataset_clean.head())

dataset_clean = dataset_clean.drop(['arrival_year','adults','children','babies','reserved_room_type','assigned_room_type'],axis=1)

X = dataset_clean.iloc[:, [0] + list(range(2, len(dataset_clean.columns)))]
y = dataset_clean.iloc[:, 1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=0)



np.random.seed(123)
rd_clf1 = RandomForestClassifier(n_estimators = 112, min_samples_split = 4)
rd_clf1.fit(X_train, y_train)

y_pred_rd_clf = rd_clf1.predict(X_test)

acc_rd_clf = accuracy_score(y_test, y_pred_rd_clf)
conf = confusion_matrix(y_test, y_pred_rd_clf)
clf_report = classification_report(y_test, y_pred_rd_clf)

print(f"Accuracy Score of Random Forest is : {acc_rd_clf}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")

# create a pickle file
import pickle
pickle_out = open("rd_clf1.pkl","wb")
pickle.dump(rd_clf1, pickle_out)
pickle_out.close()

# prediction=rd_clf.predict([[1,4.33073334,0,0,3,2,0,0,2,5,0,0,0,3,3,0,2,4.668144985,0,0,0,2]])
# print(prediction)


# Make the prediction
# prediction = rd_clf.predict([[1,3.828641396,0,1,3,3,0,0,0,6,0,0,0,3,3,0,2,4.698660529,0,1,0,3]])
# print(prediction)

