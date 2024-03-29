#Data_Preprocessing
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset


dataset=pd.read_csv('Salary_Data.csv');


#Here purchased is only the dependent variable while rest are independent
X=dataset.iloc[:,:-1].values
#Creating the dependent variable vector
y=dataset.iloc[:,1].values




#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0);


#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test results
y_pred=regressor.predict(X_test)
print(y_pred)

#Visualising the training set results
plt.scatter(X_train,y_train,color='red');
plt.plot(X_train,regressor.predict(X_train),color='blue');
plt.title('Years of Experience VS Salary(Training Set)')
plt.xlabel('Years of Experience');
plt.ylabel('Salary');
plt.show()

#Visualising the test set results
plt.scatter(X_test,y_test,color='red');
plt.plot(X_train,regressor.predict(X_train),color='blue');
plt.title('Years of Experience VS Salary(Testing Set)')
plt.xlabel('Years of Experience');
plt.ylabel('Salary');
plt.show()

















