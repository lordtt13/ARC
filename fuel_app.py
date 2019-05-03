# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:08:15 2019

@author: tanma
"""
import matplotlib.pyplot as plt
import pandas as pd,numpy as np

data = pd.read_csv('fuel.csv',low_memory = False)
data_master = pd.read_csv('fuel_master.csv')
data = data[['Quantity','Odometer','Month','District','PRV NO.','Amount','Product','Day']]

copy = data
# copy = copy.dropna()    
copy = copy[copy.Odometer != '-']
copy = copy[copy.Quantity != '-']

vehicles = sorted(list(set(copy['PRV NO.'])))
months = sorted(list(set(copy['Month'])))

max_days = []
for i in months:
    max_days.append(max(copy['Day'][copy.Month == i]))

outliers = pd.DataFrame()
odometer = []
vehicle_type = []
fuel_type = []
quantity = []
for i in vehicles:
    odometer.append(list(copy['Amount'][copy['PRV NO.'] == i].values))
    vehicle_type.append(list(data_master['Vehicle type'][data_master['PRV No.'] == i].values))
    fuel_type.append(list(copy['Product'][copy['PRV NO.'] == i].values))
    quantity.append(list(map(float,list(copy['Quantity'][copy['PRV NO.'] == i].values))))

outliers['PRV No.'] = vehicles
outliers['Vehicle Type'] = list(np.asarray(vehicle_type).flatten())
outliers['Fuel Type'] = [set(x) for x in fuel_type]
outliers['Amount Per Day'] = [sum(x)/sum(max_days) for x in odometer]
outliers['Total Amount'] = [sum(x) for x in odometer]
outliers['Quantity'] = [sum(x) for x in quantity]
outliers = outliers.dropna()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x = outliers.iloc[:,3:].values
regressor = LinearRegression()
x_hat = np.reshape(x[:,2],(-1,1))
y = np.reshape(x[:,1],(-1,1))
x_train,x_test,y_train,y_test = train_test_split(x_hat,y,test_size = 0.2)
regressor.fit(x_train,y_train)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,regressor.predict(x_test)))

plt.scatter(outliers['PRV No.'],outliers['Quantity'])
plt.title('Scatter Plot')
plt.xlabel('PRV No.')
plt.ylabel('Fuel Consumed total in Litres')
plt.show()

plt.scatter(outliers['PRV No.'],outliers['Amount Per Day'])
plt.title('Scatter Plot')
plt.xlabel('PRV No.')
plt.ylabel('Fuel Consumed per Day in INR')
plt.show()

plt.plot(outliers['Quantity'],regressor.predict(x_hat))
plt.scatter(outliers['Quantity'],outliers['Total Amount'])
plt.title('Scatter Plot')
plt.xlabel('PRV No.')
plt.ylabel('Total Fuel Consumed in INR')
plt.show()