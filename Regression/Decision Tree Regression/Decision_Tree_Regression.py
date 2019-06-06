import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset

dataset=pd.read_csv('Position_Salaries.csv');

#Here purchased is only the dependent variable while rest are independent
X=dataset.iloc[:,1:2].values
#Creating the dependent variable vector
y=dataset.iloc[:,2].values

#Adding the regressor here
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predicting the new result 
y_pred=regressor.predict((6.5))


#Visualising polynomial regression results(for higher resolution and smoother curve)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff(Decision Tree Regression Model)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()







