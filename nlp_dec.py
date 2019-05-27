# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:21:00 2019

@author: tanma
"""
# Lib Imports
import os
import re
import pandas as pd, matplotlib.pyplot as plt
from keras.initializers import VarianceScaling
from dec import DEC
from collections import Counter
from wordcloud import WordCloud
from keras.optimizers import SGD
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model

# Data Read and Sort
data = pd.read_csv("Feedback 2019.csv", header = None)

data.iloc[:,3] = pd.to_datetime(data.iloc[:,3])

data.columns = ['Serial Number','Complaint Number','Conversation','Timestamp','Noise 1','Noise 2','Noise 3']
data.sort_values('Timestamp',axis = 0,ascending = False,inplace = True)

data = data[['Conversation','Timestamp']]

data = data[:100000]

stopwords = ['q','p','ans','comment',' ans',' q',' comment',' p','do you want to share any additional feedback']

# Wordcloud make and Plot Func
def make_wordcloud(optimum,arr,a,str_lit):
    for j in range(optimum):
        n = []
        for i in arr['Remarks'][arr[a] == j]:
            n.append(i)
        
        wordcloud = WordCloud(width = 800, height = 800, 
                        background_color ='white', 
                        stopwords = stopwords, 
                        min_font_size = 10).generate(' '.join(n)) 
        
        plt.figure(figsize = (8, 8), facecolor = None) 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
        plt.savefig(str_lit + str(j), dpi = 500)  
        plt.show()

# Stopwords Removed
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

# Cluster Level 1 and 2 added to DataFrame
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

del a,b,c

# Data made for Model
def remarks(arr):
    med = []
    for i in arr:
        review = ' '.join(i)
        review = review.split()
        if('remark' in review):
            med.append(' '.join(review[review.index('remark')+1:]))
        else:
            med.append('remark not given')
    return med

copy['Remarks'] = remarks(copy['Conversation'])

def preprocess(literal,Level,arr):
    copy_actual = arr[arr[literal] == Level]
    
    actual_words = []
    for i in copy_actual['Remarks']:
        review = i.split()
        for j in review:
            actual_words.append(j)
    
    counts = Counter(actual_words)
    
    actual_stopwords = []
    for i,j in zip(counts,counts.values()):
        if(j < 6):
            actual_stopwords.append(i)
        elif(len(list(i)) < 3):
            actual_stopwords.append(i)
            
    actual_split = []
    for i in copy_actual['Remarks']:
        review = i.split()
        review = [word for word in review if not word in actual_stopwords]
        actual_split.append(' '.join(review))
    
    copy_actual['Remarks'] = actual_split
    copy_actual = copy_actual[copy_actual['Remarks'] != '']
    
    while '' in actual_split:
        actual_split.remove('')
    
    cv = CountVectorizer()
    X = cv.fit_transform(actual_split).toarray()
#    V = cv.vocabulary_
#    B = cv.get_feature_names()
        
    return X,counts,copy_actual

def preprocess_with_threshold(literal,Level,arr,min_thresh,max_thresh):
    copy_actual = arr[arr[literal] == Level]
    
    actual_words = []
    for i in copy_actual['Remarks']:
        review = i.split()
        for j in review:
            actual_words.append(j)
    
    counts = Counter(actual_words)
    
    actual_stopwords = []
    for i,j in zip(counts,counts.values()):
        if(j < min_thresh or j > max_thresh):
            actual_stopwords.append(i)
        elif(len(list(i)) < 3):
            actual_stopwords.append(i)
            
    actual_split = []
    for i in copy_actual['Remarks']:
        review = i.split()
        review = [word for word in review if not word in actual_stopwords]
        actual_split.append(' '.join(review))
    
    copy_actual['Remarks'] = actual_split
    copy_actual = copy_actual[copy_actual['Remarks'] != '']
    
    while '' in actual_split:
        actual_split.remove('')
    
    cv = CountVectorizer()
    X = cv.fit_transform(actual_split).toarray()
#    V = cv.vocabulary_
#    B = cv.get_feature_names()
        
    return X,counts,copy_actual

def elbow(X,literal):
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init =10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1,11),wcss)
    plt.title('Elbow graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') 
    plt.savefig('Intial Cluster WCSS '+ literal +'.png', dpi = 500)
    plt.show()

X,counts,copy_actual = preprocess('Cluster Level 2',1,copy)

# elbow(X,'Satisfied DEC')

# Hyperparameters Definition
init = 'glorot_uniform'
pretrain_optimizer = 'nadam'
batch_size = 512
maxiter = 2e4
tol = 0.001
save_dir = 'results'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

update_interval = 50
pretrain_epochs = 100
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                       distribution='uniform')  
# [-limit, limit], limit=sqrt(1./fan_in)
#pretrain_optimizer = SGD(lr=1, momentum=0.9)

# Pretraining and Training
dec = DEC(dims=[X.shape[-1], 500, 500, 2000, 100], n_clusters=4, init=init)

dec.pretrain(x=X, optimizer=pretrain_optimizer,
             epochs=pretrain_epochs, batch_size=batch_size,
             save_dir=save_dir)

dec.model.summary()

dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')

y_pred = dec.fit(X, tol=tol, maxiter=maxiter, batch_size=batch_size,
                 update_interval=update_interval, save_dir=save_dir)

copy_actual['Cluster Level 3'] = list(y_pred)

copy_actual.to_csv('Satisfied Clustering.csv', encoding='utf-8', index=False)

make_wordcloud(4,copy_actual,'Cluster Level 3','Satisfied DEC Cluster')

X_1,counts_not,copy_actual_not = preprocess('Cluster Level 2',2,copy)

#elbow(X_1,'Unsatisfied DEC')

dec_not = DEC(dims=[X_1.shape[-1], 500, 500, 2000, 100], n_clusters=2, init=init)

dec_not.pretrain(x=X_1, optimizer=pretrain_optimizer,
             epochs=pretrain_epochs, batch_size=batch_size,
             save_dir=save_dir)

dec_not.model.summary()

dec_not.compile(optimizer=SGD(0.01, 0.9), loss='kld')

y_pred_not = dec_not.fit(X_1, tol=tol, maxiter=maxiter, batch_size=batch_size,
                 update_interval=update_interval, save_dir=save_dir)

copy_actual_not['Cluster Level 3'] = list(y_pred_not)

copy_actual_not.to_csv('Not Satisfied Clustering.csv', encoding='utf-8', index=False)
        
make_wordcloud(2,copy_actual_not,'Cluster Level 3','Not Satisfied DEC Cluster')

X_0_0,counts_0,copy_actual_0 = preprocess_with_threshold('Cluster Level 3',0,copy_actual,1,1000)

dec_0_0 = DEC(dims=[X_0_0.shape[-1], 500, 500, 2000, 100], n_clusters=2, init=init)

dec_0_0.pretrain(x=X_0_0, optimizer=pretrain_optimizer,
             epochs=pretrain_epochs, batch_size=batch_size,
             save_dir=save_dir)

dec_0_0.model.summary()

dec_0_0.compile(optimizer=SGD(0.01, 0.9), loss='kld')

y_pred_0_0 = dec_0_0.fit(X_0_0, tol=tol, maxiter=maxiter, batch_size=batch_size,
                 update_interval=update_interval, save_dir=save_dir)

copy_actual_0['Cluster Level 4'] = list(y_pred_0_0)

copy_actual_0.to_csv('Satisfied Clustering First Subcluster 0.csv', encoding='utf-8', index=False)
        
make_wordcloud(2,copy_actual_0,'Cluster Level 4','Satisfied DEC Cluster First Subcluster 0')