#!/usr/bin/env python
# coding: utf-8

# In[7]:


from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')
#Importing the dataset
data = pd.read_csv('E:\Digitalent\Latihan big data\Data Latih/xclara.csv')
print("Input Data and Shape")
print("data.shape")
data.head()
k=17
kmeans = KMeans(n_clusters=k).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
colors=['#00FF00']
fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:,1], c= kmeans.labels_.astype('float64'), s=200, alpha=0.5)
ax.scatter(centroids[:, 0], centroids[:, 1],marker='*', c='#050505', s=200)


# In[ ]:




