#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('Position_Salaries.csv');
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


#We  need to apply feature scaling to this model 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)


#Predicting the  result 
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array((  [6.5] ))))


#Visualising polynomial regression results
#X_grid=np.arange(min(X),max(X),0.1)
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff(SVR Model)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()