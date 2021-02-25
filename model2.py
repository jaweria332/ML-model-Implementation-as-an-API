"""This is a very basic Linear Regression model trained for the implementation as an API as Flask App"""


#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#importing dataset
dataset = pd.read_csv("E:\\ML Zero to Hero\\headbrain.csv")
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


#Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)


#Training our model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)



#Drawing predictions
predictions = model.predict(X_test)



#Finding score
from sklearn.metrics import accuracy_score
acc = accuracy_score(predictions, Y_test)
print("Accuracy = ", round(acc * 100))



#Storing trained model in pickle file
import pickle
pickle_out = open("classifier2.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()