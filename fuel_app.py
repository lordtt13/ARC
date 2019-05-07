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
data = data.fillna(0)


copy = data[data.District == 'Varanasi']
# copy = copy.dropna()    
copy = copy[copy.Odometer != '-']
copy = copy[copy.Quantity != '-']

vehicles = sorted(list(set(copy['PRV NO.'])))
months = sorted(list(set(copy['Month'])))

max_days = []
for i in months:
    max_days.append(max(copy['Day'][copy.Month == i]))

copy_one = pd.DataFrame()
odometer = []
odometer_data = []
vehicle_type = []
fuel_type = []
quantity = []
for i in vehicles:
    odometer.append(list(map(float,list(copy['Amount'][copy['PRV NO.'] == i].values))))
    vehicle_type.append(list(data_master['Vehicle type'][data_master['PRV No.'] == i].values))
    fuel_type.append(list(copy['Product'][copy['PRV NO.'] == i].values))
    quantity.append(list(map(float,list(copy['Quantity'][copy['PRV NO.'] == i].values))))
    odometer_data.append(list(copy['Odometer'][copy['PRV NO.'] == i].values))
        
copy_one['PRV No.'] = vehicles
copy_one['Vehicle Type'] = list(np.asarray(vehicle_type).flatten())
copy_one['Fuel Type'] = [set(x) for x in fuel_type]
copy_one['Amount Per Day'] = [sum(x)/sum(max_days) for x in odometer]
copy_one['Total Amount'] = [sum(x) for x in odometer]
copy_one['Quantity'] = [sum(x) for x in quantity]
copy_one["Odometer Data"] = odometer_data
copy_one = copy_one.fillna(0)

copy_double = []
vehicles = []
for i,j in enumerate(copy_one['Odometer Data']):
    if (j.count(0) < 0.5*len(j)):
        copy_double.append(j)
        vehicles.append(copy_one.iloc[i,0])
      
copy_triple = []        
for i in copy_double:
    diff = [int(i[n]) - int(i[n-1]) for n in range(1,len(i))]
    copy_triple.append(diff)

copy_fourth = []
for i in copy_triple:
    diff = [j for j in i if(j > 0 and j <1000)]
    copy_fourth.append(diff)
            
copy_fifth = []    
for i in copy_fourth:
    if(len(i) > 9):
        copy_fifth.append(i)

copy_sixth = []
for i in copy_fifth:
    for j in i:
        copy_sixth.append(j)
        
mean_odo = int(np.mean(copy_sixth))

def threshold(arr,mean_arr):
    j = 0.01
    excess = []
    while(len(excess) < 0.75*len(arr)):
        excess = [i for i in arr if(i > mean_arr - j*mean_arr and i < mean_arr + j*mean_arr)]
        j += 0.01
    return min(excess),max(excess)
    
min_amt,max_amt = threshold(copy_one['Amount Per Day'],np.mean(copy_one['Amount Per Day']))
min_fuel,max_fuel = threshold(copy_one['Quantity'],np.mean(copy_one['Quantity']))
        
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x = copy_one.iloc[:,3:].values
regressor = LinearRegression()
x_hat = np.reshape(x[:,2],(-1,1))
y = np.reshape(x[:,1],(-1,1))
x_train,x_test,y_train,y_test = train_test_split(x_hat,y,test_size = 0.2)
regressor.fit(x_train,y_train)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,regressor.predict(x_test)))

plt.scatter(copy_one['PRV No.'],copy_one['Quantity'])
plt.title('Scatter Plot')
plt.xlabel('PRV No.')
plt.ylabel('Fuel Consumed total in Litres')
plt.show()

plt.scatter(copy_one['PRV No.'],copy_one['Amount Per Day'])
plt.title('Scatter Plot')
plt.xlabel('PRV No.')
plt.ylabel('Fuel Consumed per Day in INR')
plt.show()

plt.plot(copy_one['Quantity'],regressor.predict(x_hat))
plt.scatter(copy_one['Quantity'],copy_one['Total Amount'])
plt.title('Scatter Plot')
plt.xlabel('PRV No.')
plt.ylabel('Total Fuel Consumed in INR')
plt.show()