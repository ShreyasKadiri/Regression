#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset

dataset=pd.read_csv('50_Startups.csv');

#Here purchased is only the dependent variable while rest are independent
X=dataset.iloc[:,: -1].values
#Creating the dependent variable vector
y=dataset.iloc[:,4].values

#Categorical data,countries and purchased are categorical data since they have france,germany,spain and yes ,no as categories
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder();


#Since elements like california,new york, are in text we have to encode it
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#To avoid the dummy variable trap,the below statement indicates that the first column with the index 0 in python is not considered 
X=X[:,1:]

#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0);

#Fitting the Multiple Linear Regression Model into the training_set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)


#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#0.99 is the highest predicted value,so we remove that as per our backward elimination algorithm,now we remove X2
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()






