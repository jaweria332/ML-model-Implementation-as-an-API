#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


#importing dataset
dataset = pd.read_csv("E:\\ML Zero to Hero\\headbrain.csv")
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]



#training the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


#finding accuracy
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(X,Y)
MSE = math.sqrt(MSE)


#finding the accuracy
acc = LR.score(X,Y)
print("Mean squared error = " + str(MSE))
print("Accuracy = " + str(acc))



#Storing trained model in pickle file
import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(LR, pickle_out )
pickle_out.close()


#predict result using the trained model
Headsize = int(input("Enter Head Size (in cm^3)"))
brainweight=LR.predict([[Headsize]])
print("Prediced Brain Weight(grams) = ", brainweight)