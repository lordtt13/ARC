# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:35:08 2019

@author: tanma
"""

import re
import pandas as pd, matplotlib.pyplot as plt

data = pd.read_csv("Feedback 2019.csv", header = None)

data.iloc[:,3] = pd.to_datetime(data.iloc[:,3])

data.columns = ['Serial Number','Complaint Number','Conversation','Timestamp','Noise 1','Noise 2','Noise 3']
data.sort_values('Timestamp',axis = 0,ascending = False,inplace = True)

data = data[:20000]

stopwords = ['q','p','ans','remark','comment',' ans',' q',' remark',' comment',' p']

med = []
for i in data['Conversation']:
    review = re.sub('[^a-zA-Z]', ' ', str(i))
    review = review.lower()
    review = review.split('  ')
    review = [i for i in review if not i in stopwords]
    review = [i for i in review if (len(i)!=0)]
    med.append(review[1:])

copy = data

copy['Conversation'] = med 

a = []
b = []
for i in copy['Conversation']:
    if (i[0] == 'yes' and len(i) > 2):
        a.append(1)
        b.append(i[2])
    else:
        a.append(0)
        b.append('response not given')

copy['Satisfaction Feedback']  = b        
copy['Cluster Level 1'] = a
