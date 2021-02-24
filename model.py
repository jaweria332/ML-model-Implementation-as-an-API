#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


#Importing dataset
dataset = pd.read_csv("E:\\ML Zero to Hero\\BankNote_Authentication.csv")


#Extracting features and target from dataset
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


#Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3, random_state=0)


#Training our model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, Y_train)


#Predicting on test set
y_pred = classifier.predict(X_test)


#Finding accuracy score
from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test, y_pred)
print("Score = ", (round(score * 100)))


#Create a pickle using serialization
import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out )
pickle_out.close()


#Drawing predictions
classifier.predict([[2,3,4,5]])