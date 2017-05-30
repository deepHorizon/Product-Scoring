# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:28:28 2017

@author: Shaurya Rawat
"""

import pandas as pd

#Read the Data 
data=pd.read_csv("D:\Jobs\Stylight Startup\product_scoring.csv")

# Inspect the data
data.head(5)
data.tail(5)

#We have 5 columns with product_id, category_id, cpc(cost per click), clicks and views
# Views has some null values
# We will assume that these records dont have any views and put them as 0
data['views'].fillna(0,inplace=True)
#Lets check if we have  succeeded
data['views'].head(5)
# We also check if there is any other missing value in the data
data.isnull().any()
# No missing value in data now

#Now we check the descriptive statistics of the data
data.describe()
# Product and cateogry are irrelevant from a statistical perspective
# cpc min: 1.0942 max: 2.599 mean: 1.505468
# clicks min: 0.00 max: 1095 mean: 2.3765
# views min: 0.00 max: 32168 mean: 116.8306

#let us check how many different products we have
data['product_id'].unique()
# All products are unique

## For the product scoring we need to consider two main attributes
# Total money spent on clicks and ctr(click through rate)
# Let us create two new columns for the same
data['revenue']=data['clicks']*data['cpc']
data['revenue'].describe()
# min: 0.0, max: 1898.088, mean: 4.0621
data['ctr']=(data['clicks']/data['views'])*100
data['ctr']
# ctr column has some NA values because of division by records where views=0
# we replace these with 0
data['ctr'].fillna(0,inplace=True)
data['ctr'].describe()
# min: 0.00, max: 40.00, mean: 0.5855

#check correlation
data.corr()
# correlation between the attributes especially clicks,view,cpc and ctr and total_money_spent
# are as expected

## Now as we have a cpc campaign in this, we will delete the rows where clicks=0 as it doesnt
# contribute towards our scoring and it gives 0 value to ctr and total_money_spent
data.shape # 10000,7
data=data.drop(data[data.clicks==0].index)
data.shape # 1762,7
data.describe()
# descriptive statistics change because of we dropped rows that we did not need
data1=data.drop(['product_id','category_id'],axis=1)
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=6,random_state=0).fit(data1)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pylab as pl
pca=PCA(n_components=2).fit(data1)
pca_2d=pca.transform(data1)
pl.figure('K means with 6 clusters')
pl.scatter(pca_2d[:,0],pca_2d[:,1],c=kmeans.labels_)
pl.show()

#####
x=[i for i in data['ctr'].values if i<6.0]
len(x) #1497
data2=data1.loc[data1['ctr']<6.0]
data2=data2.drop(['clicks','views','cpc'],axis=1)
# data with ctr less than 6%
kmeans=KMeans(n_clusters=5,random_state=0).fit(data2)
pca=PCA(n_components=2).fit(data2)
pca_2d=pca.transform(data2)
pl.figure('K Means with 6 Clusters: CTR<6.0')
pl.scatter(pca_2d[:,0],pca_2d[:,1],c=kmeans.labels_)
pl.show()

### Clustering 
#import seaborn as sns
#import time
#import numpy as np
#import sklearn.cluster as cluster
#sns.set_context('poster')
#sns.set_color_codes()
#plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
#
#def plot_clusters(data, algorithm, args, kwds):
#    start_time = time.time()
#    labels = algorithm(*args, **kwds).fit_predict(data)
#    end_time = time.time()
#    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
#    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
#    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
#    frame = plt.gca()
#    frame.axes.get_xaxis().set_visible(False)
#    frame.axes.get_yaxis().set_visible(False)
#    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
#    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
#
#plot_clusters(data2,cluster.AffinityPropagation,(),{'preference':-5.0,'damping':0.95})
#plot_clusters(data2, cluster.MeanShift, (0.175,), {'cluster_all':False})
#plot_clusters(data2, cluster.KMeans, (), {'n_clusters':6})

### Clustering using hdbscan
import hdbscan
clusterer=hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels=clusterer.fit_predict(data2)
pl.figure('HDBSCAN Clustering')
pl.scatter(pca_2d[:,0],pca_2d[:,1],c=cluster_labels)
pl.show()

##
import scipy.stats as ss
data2=data.drop(['product_id','category_id'],axis=1)
data2=data2.drop(['revenue','ctr'],axis=1)
rank=ss.rankdata(data2)
data2.describe()

##### Logistic Approach
data['ctr']=(data['clicks']/data['views'])*100
data['ctr'].fillna(0,inplace=True)

# ctr benchmark 
# min 0 - 40 max
#good ctr 1.5 - 3
# 3- 6
# 6-10
data.sort(['ctr'],ascending=False)
data['score']=5*data['clicks']+3*data['views']+data['ctr']
data.sort(['score'],ascending=False)
submission=pd.DataFrame({"product_id":data['product_id'],"Score":data['score']})
submission.to_csv('Shaurya_Rawat_Stylight_Task.csv',index=False)
