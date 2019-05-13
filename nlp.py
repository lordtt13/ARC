# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:53:22 2019

@author: tanma
"""
import re
import pandas as pd, matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

data = pd.read_csv("Feedback 2019.csv", header = None)

data = data[:10000]

med = []
for i in data.iloc[:,2].values:
    review = re.sub('[^a-zA-Z]', ' ', str(i))
    review = review.lower()
    review = review.split()
    for j in review:
        if j == 'remark':
            med.append(' '.join(review[review.index(j)+2:]))

actual = []    
for i in med:
    if(len(i) != 0):
        actual.append(i)

corpus = []
c = ''
for i in actual:
    review = i.split()
    review = [word for word in review if not word in set(stopwords.words('up100 2'))]
    review = ' '.join(review)
    corpus.append(review)
    c += ' ' + review
    
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
V = cv.vocabulary_
B = cv.get_feature_names()

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init =10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.savefig('Intial Cluster WCSS.png', dpi = 500)
plt.show()

kmeans = KMeans(n_clusters = 7 , init = 'k-means++', max_iter = 300, n_init = 10)
y_kmeans = kmeans.fit_predict(X)
yp = pd.DataFrame(y_kmeans)
X = pd.DataFrame(X)
actual = pd.DataFrame(actual)

frames = [actual, yp, X]
fr = pd.concat(frames, axis =1, names = ['Remarks','Cluster'], 
               sort=False, ignore_index =True)

fr_new = fr[fr.iloc[:,1] == 3]
fr_new = fr_new.iloc[:,2:].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Word Indices')
plt.ylabel('Euclidean distances')
plt.savefig('Initial Dendrogram.png', dpi = 500)
plt.show()

dendrogram = sch.dendrogram(sch.linkage(fr_new, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Word Indices')
plt.ylabel('Euclidean distances')
plt.savefig('0 Dendrogram.png', dpi = 500)
plt.show()

Xcess = fr_new
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(Xcess)

fr_new_one = fr[fr.iloc[:,1] == 0]
fr_new_one = fr.iloc[:,0]
y_hc = pd.DataFrame(y_hc)

frames_one = [fr_new_one, y_hc]
fr_new = pd.concat(frames_one, axis =1, names = ['Remarks','Cluster'], 
               sort=False, ignore_index =True)