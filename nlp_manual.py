# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:35:08 2019

@author: tanma
"""

import re
import pandas as pd, matplotlib.pyplot as plt,numpy as np
from collections import Counter
from wordcloud import WordCloud

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
    
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
X = cv.fit_transform(actual_split).toarray()
V = cv.vocabulary_
B = cv.get_feature_names()

from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
X = pca.fit_transform(X)

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init =10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.savefig('Satisfied Initial Cluster WCSS.png', dpi = 500)
plt.show()

opt = 5

kmeans = KMeans(n_clusters = opt, init = 'k-means++', max_iter = 300, n_init = 10)
y_kmeans = kmeans.fit_predict(X)

copy_actual['Cluster Level 3'] = list(y_kmeans)

for j in range(opt):
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
    plt.savefig('Satisfied WordCloud for Cluster '+str(j), dpi = 500)  
    plt.show() 

copy_actual.to_csv('Satisfied Clustering.csv', encoding='utf-8', index=False)

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
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init =10)
    kmeans.fit(X_1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.savefig('Not Satisfied Initial Cluster WCSS.png', dpi = 500)
plt.show()

kmeans_not = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10)
y_kmeans_not = kmeans_not.fit_predict(X_1)

copy_actual_not['Cluster Level 3'] = list(y_kmeans_not)

for j in range(2):
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

copy_actual.to_csv('Not Satisfied Clustering.csv', encoding='utf-8', index=False)

copy_not_1 = copy_actual_not[copy_actual_not['Cluster Level 3'] == 1]

actual_not_1 = []
for i in copy_not_1['Conversation']:
    review = i.split()
    for j in review:
        actual_not_1.append(j)

counts_not_1 = Counter(actual_not_1)

actual_stopwords_not_1 = []
for i,j in zip(counts_not_1,counts_not_1.values()):
    if(j < 2 or  j > 1000):
        actual_stopwords_not_1.append(i)
        
actual_split_not_1 = []
for i in copy_not_1['Conversation']:
    review = i.split()
    review = [word for word in review if not word in actual_stopwords_not_1]
    actual_split_not_1.append(' '.join(review))

copy_not_1['Conversation'] = actual_split_not_1
copy_not_1 = copy_not_1[copy_not_1['Conversation'] != '']

while '' in actual_split_not_1:
    actual_split_not_1.remove('')

X_1 = cv.fit_transform(actual_split_not_1).toarray()
V_1 = cv.vocabulary_
B_1 = cv.get_feature_names()
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init =10)
    kmeans.fit(X_1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.savefig('Not Satisfied 1 Level 3 Cluster WCSS.png', dpi = 500)
plt.show()

kmeans_not_1 = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 10)
y_kmeans_not_1 = kmeans_not_1.fit_predict(X_1)

copy_not_1['Cluster Level 4'] = list(y_kmeans_not_1)

for j in range(6):
    n = []
    for i in copy_not_1['Conversation'][copy_not_1['Cluster Level 4'] == j]:
        n.append(i)
    
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(' '.join(n)) 
    
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.savefig('Not Satisfied 1 Level 3 WordCloud for Cluster '+str(j), dpi = 500)  
    plt.show()     