import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df=pd.read_csv("D:/Data Science/FuelConsumptionCo2.csv")




cdf=df[['ENGINESIZE','CO2EMISSIONS']]

split=np.random.rand(len(cdf))<0.8
train=cdf[split]
test=cdf[~split]


x_train=train[['ENGINESIZE']].values
y_train=train[['CO2EMISSIONS']].values

x_test=test[['ENGINESIZE']].values
y_test=test[['CO2EMISSIONS']].values

lr=linear_model.LinearRegression().fit(x_train,y_train)

y_pred=lr.predict(x_test)



mse=mean_squared_error(y_test,y_pred)

print("Coefficient + ",lr.coef_,"\nIntercept = ",lr.intercept_,
      "\nMean squared error= ",mse)

plt.scatter(x_test[:, 0], y_test, color='blue', label="Actual") 
plt.plot(x_test[:, 0], y_pred, color='red', label="Predicted")  
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.legend()
plt.show()




