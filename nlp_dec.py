# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:21:00 2019

@author: tanma
"""

import re
import pandas as pd, matplotlib.pyplot as plt,numpy as np
from keras.initializers import VarianceScaling
from dec import DEC
from collections import Counter
from wordcloud import WordCloud
from keras.optimizers import SGD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


data = pd.read_csv("Feedback 2019.csv", header = None)

data.iloc[:,3] = pd.to_datetime(data.iloc[:,3])

data.columns = ['Serial Number','Complaint Number','Conversation','Timestamp','Noise 1','Noise 2','Noise 3']
data.sort_values('Timestamp',axis = 0,ascending = False,inplace = True)

data = data[['Conversation','Timestamp']]

data = data[:100000]

stopwords = ['q','p','ans','remark','comment',' ans',' q',' remark',' comment',' p','do you want to share any additional feedback']

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

c = []
for i in copy['Satisfaction Feedback']:
    if(i == 'satisfied' or i == 'very satisfied'):
        c.append(1)
    elif(i == 'not satisfied'):
        c.append(2)
    elif(i == 'not aware'):
        c.append(3)
    else:
        c.append(0)

copy['Cluster Level 2'] = c
copy = copy[copy['Satisfaction Feedback'] != ' ']

copy_actual = copy[copy['Cluster Level 2'] == 1]

actual = []
for i in copy_actual['Conversation']:
    review = i[4:]
    review = ' '.join(review)
    actual.append(review)

copy_actual['Conversation'] = actual

actual_words = []
for i in actual:
    review = i.split()
    for j in review:
        actual_words.append(j)

counts = Counter(actual_words)

actual_stopwords = []
for i,j in zip(counts,counts.values()):
    if(j < 2 or j > 1000):
        actual_stopwords.append(i)
        
actual_split = []
for i in copy_actual['Conversation']:
    review = i.split()
    review = [word for word in review if not word in actual_stopwords]
    actual_split.append(' '.join(review))

copy_actual['Conversation'] = actual_split
copy_actual = copy_actual[copy_actual['Conversation'] != '']

while '' in actual_split:
    actual_split.remove('')
    

cv = CountVectorizer()
X = cv.fit_transform(actual_split).toarray()
V = cv.vocabulary_
B = cv.get_feature_names()

init = 'glorot_uniform'
pretrain_optimizer = 'adam'
dataset = 'mnist'
batch_size = 128
maxiter = 2e4
tol = 0.001
save_dir = 'results'

import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

update_interval = 50
pretrain_epochs = 100
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                       distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
#pretrain_optimizer = SGD(lr=1, momentum=0.9)

dec = DEC(dims=[X.shape[-1], 500, 500, 2000, 10], n_clusters=3, init=init)

dec.pretrain(x=X, optimizer=pretrain_optimizer,
             epochs=pretrain_epochs, batch_size=batch_size,
             save_dir=save_dir)

dec.model.summary()

dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')

y_pred = dec.fit(X, tol=tol, maxiter=maxiter, batch_size=batch_size,
                 update_interval=update_interval, save_dir=save_dir)

copy_actual['Cluster Level 3'] = list(y_pred)

for j in range(3):
    n = []
    for i in copy_actual['Conversation'][copy_actual['Cluster Level 3'] == j]:
        n.append(i)
    
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(' '.join(n)) 
    
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.savefig('Satisfied DEC WordCloud for Cluster '+str(j), dpi = 500)  
    plt.show()

copy_actual_not = copy[copy['Cluster Level 2'] == 2]

actual_not = []
for i in copy_actual_not['Conversation']:
    review = i[4:]
    review = ' '.join(review)
    actual_not.append(review)

copy_actual_not['Conversation'] = actual_not

actual_words_not = []
for i in actual_not:
    review = i.split()
    for j in review:
        actual_words_not.append(j)

counts_not = Counter(actual_words_not)

actual_stopwords_not = []
for i,j in zip(counts_not,counts_not.values()):
    if(j < 2 or j > 1000):
        actual_stopwords_not.append(i)
        
actual_split_not = []
for i in copy_actual_not['Conversation']:
    review = i.split()
    review = [word for word in review if not word in actual_stopwords_not]
    actual_split_not.append(' '.join(review))

copy_actual_not['Conversation'] = actual_split_not
copy_actual_not = copy_actual_not[copy_actual_not['Conversation'] != '']

while '' in actual_split_not:
    actual_split_not.remove('')

X_1 = cv.fit_transform(actual_split_not).toarray()
V_1 = cv.vocabulary_
B_1 = cv.get_feature_names()

dec_not = DEC(dims=[X_1.shape[-1], 500, 500, 2000, 10], n_clusters=3, init=init)

dec_not.pretrain(x=X_1, optimizer=pretrain_optimizer,
             epochs=pretrain_epochs, batch_size=batch_size,
             save_dir=save_dir)

dec_not.model.summary()

dec_not.compile(optimizer=SGD(0.01, 0.9), loss='kld')

y_pred_not = dec_not.fit(X_1, tol=tol, maxiter=maxiter, batch_size=batch_size,
                 update_interval=update_interval, save_dir=save_dir)

copy_actual_not['Cluster Level 3'] = list(y_pred_not)

for j in range(3):
    n = []
    for i in copy_actual_not['Conversation'][copy_actual_not['Cluster Level 3'] == j]:
        n.append(i)
    
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(' '.join(n)) 
    
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.savefig('Not Satisfied WordCloud for Cluster '+str(j), dpi = 500)  
    plt.show()