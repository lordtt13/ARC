# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:08:15 2019

@author: tanma
"""
import matplotlib.pyplot as plt
import pandas as pd,numpy as np

data = pd.read_csv('fuel.csv',low_memory = False)
data_master = pd.read_csv('fuel_master.csv')
data_master = data_master.drop_duplicates()
data = data.drop_duplicates()
data = data[data.Odometer != '-']
data = data[data.Quantity != '-']
data = data[['Quantity','Odometer','Month','District','PRV NO.','Amount','Product','Day','Txn Date']]
data['Txn Date'] = pd.to_datetime(data['Txn Date'])
data = data.fillna(0)
data = data[data.Odometer != '\n']
data.sort_values(['PRV NO.','Txn Date'],axis = 0,ascending = True,inplace = True)

copy = data

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

hexa_comp = []
for i in odometer_data:
    hexa = [int(j) for j in i]
    hexa_comp.append(hexa)

copy_one['PRV No.'] = vehicles
copy_one['Vehicle Type'] = list(np.asarray(vehicle_type).flatten())
copy_one['Fuel Type'] = [set(x) for x in fuel_type]
copy_one['Amount Per Day'] = [sum(x)/sum(max_days) for x in odometer]
copy_one['Total Amount'] = [sum(x) for x in odometer]
copy_one['Quantity'] = [sum(x) for x in quantity]
copy_one["Odometer Data"] = hexa_comp
copy_one['Total Distance Travelled'] = [max(i) - min(i) for i in hexa_comp]
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
    diff = [j for j in i if(j > 0 and j < 1000)]
    copy_fourth.append(diff)
            
copy_fifth = []    
for i in copy_fourth:
    if(len(i) > 9):
        copy_fifth.append(i)

copy_sixth = []
for i in copy_fifth:
    copy_sixth.append(sum(i))
        
mean_odo = int(np.mean(copy_sixth))

def threshold(arr,mean_arr):
    j = 0.01
    excess = []
    while(len(excess) < 0.75*len(arr)):
        excess = [i for i in arr if(i > mean_arr - j*mean_arr and i < mean_arr + j*mean_arr)]
        j += 0.01
    return min(excess),max(excess)
    
min_amt,max_amt = threshold(copy_one['Amount Per Day'],copy_one['Amount Per Day'].mean())
min_fuel,max_fuel = threshold(copy_one['Quantity'],copy_one['Quantity'].mean())
min_dis,max_dis = threshold(copy_one['Total Distance Travelled'],copy_one['Total Distance Travelled'].mean())

is_outlier_amt = []
for i in copy_one['Amount Per Day']:
    if (i < min_amt or i > max_amt):
        is_outlier_amt.append(1)
    else:
        is_outlier_amt.append(0)

is_outlier_quant = []
for i in copy_one['Quantity']:
    if (i < min_fuel or i > max_fuel):
        is_outlier_quant.append(1)
    else:
        is_outlier_quant.append(0)

is_outlier_dis = []
for i in copy_one['Total Distance Travelled']:
    if (i < min_dis or i > max_dis):
        is_outlier_dis.append(1)
    else:
        is_outlier_dis.append(0)
        
is_outlier_fuel = []
for i in copy_one['Fuel Type']:
    if (len(i) > 1):
        is_outlier_fuel.append(1)
    else:
        is_outlier_fuel.append(0)

defaulter = []
for i in copy_one['Odometer Data']:
    defaulter.append(i.count(0))

diff_defaulter = []
for i in copy_one['Odometer Data']:
    diff = [int(i[n]) - int(i[n-1]) for n in range(1,len(i))]
    x_new = [j for j in diff if(j < 1 or j > 1000)]
    diff_defaulter.append(len(x_new))

copy_one['Fuel Outlier'] = is_outlier_fuel
copy_one['Distance Difference Outlier'] = diff_defaulter
copy_one['Odometer Entry Defaulter'] = defaulter
copy_one['Distance Outlier'] = is_outlier_dis
copy_one['Quantity Outlier'] = is_outlier_quant
copy_one['Amount Outlier'] = is_outlier_amt
copy_one['Is Outlier'] = is_outlier_fuel

zipped = zip(is_outlier_fuel, diff_defaulter, defaulter, is_outlier_dis, is_outlier_quant, is_outlier_amt, is_outlier_fuel)

copy_one['Is Outlier'] = [sum(item) for item in zipped]
     
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

from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint

x_data_use = copy_one[['Amount Per Day','Quantity','Total Distance Travelled','Fuel Outlier','Distance Difference Outlier','Odometer Entry Defaulter']].values
y_data_use = copy_one.iloc[:,-1].values

input_ = Input(shape = (6,))
reg = Dense(256, activation = 'relu')(input_)
reg = Dense(128, activation = 'relu')(reg)
reg = Dense(64, activation = 'relu')(reg)
reg = Dense(32, activation = 'relu')(reg)
reg = Dense(16, activation = 'relu')(reg)
reg = Dense(8, activation = 'relu')(reg)
reg = Dense(4, activation = 'relu')(reg)
reg = Dense(2, activation = 'relu')(reg)
output = Dense(1,activation = 'relu')(reg)

model = Model(input_,output)

model.compile(optimizer = 'nadam', loss = 'mse')
filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks=[checkpoint]

model.fit(x_data_use,y_data_use,epochs = 100,batch_size = 10,callbacks = callbacks, validation_split = 0.25)

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

plt.scatter(copy_one['PRV No.'],copy_one['Total Distance Travelled'])
plt.ylim([0,1e+06])
plt.title('Scatter Plot')
plt.xlabel('PRV No.')
plt.ylabel('Total Distance Travelled')
plt.show()

plt.plot(copy_one['Quantity'],regressor.predict(x_hat))
plt.scatter(copy_one['Quantity'],copy_one['Total Amount'])
plt.title('Scatter Plot')
plt.xlabel('PRV No.')
plt.ylabel('Total Fuel Consumed in INR')
plt.show()