# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:26:58 2018

@author: danie
"""

from scipy.io import loadmat
import numpy as np
import json
from collections import Counter
from scipy.spatial import distance
from sklearn.decomposition import PCA, KernelPCA
import time
from sklearn.cluster import KMeans

#%% load "cuhk03_new_protocol_config_labeled.mat"

all_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')

camId = all_idxs['camId'].flatten()
filelist = all_idxs['filelist'].flatten()
labels = all_idxs['labels'].flatten()

train_idx = all_idxs['train_idx'].flatten() - 1 #subtract one to go from Matlab indexing to Python indexing
gallery_idx = all_idxs['gallery_idx'].flatten() - 1
query_idx = all_idxs['query_idx'].flatten() - 1

#%% load the features matrix and convert it to a np array

with open('feature_data.json', 'r') as f:
 features = json.load(f)
 
features = np.asarray(features)

#%% split the features matrix in training, gallery and query matrices

features_train = np.zeros((len(train_idx), len(features[0])))
labels_train = np.zeros(len(train_idx))
camId_train = np.zeros(len(train_idx))

for i in range(len(train_idx)):
    features_train[i,:] = features[train_idx[i],:]
    labels_train[i] = labels[train_idx[i]]
    camId_train[i] = camId[train_idx[i]]
    
    
features_gallery = np.zeros((len(gallery_idx), len(features[0])))
labels_gallery = np.zeros(len(gallery_idx))
camId_gallery = np.zeros(len(gallery_idx))

for i in range(len(gallery_idx)):
    features_gallery[i,:] = features[gallery_idx[i],:]
    labels_gallery[i] = labels[gallery_idx[i]]
    camId_gallery[i] = camId[gallery_idx[i]]
    
    
features_query = np.zeros((len(query_idx), len(features[0])))
labels_query = np.zeros(len(query_idx))
camId_query = np.zeros(len(query_idx))

for i in range(len(query_idx)):
    features_query[i,:] = features[query_idx[i],:]
    labels_query[i] = labels[query_idx[i]]
    camId_query[i] = camId[query_idx[i]]

#%% split the training data in train_query and train_gallery
    
count_query = 0
count_gallery = 0
train_gallery = np.zeros((5836, 2048))
train_gallery_labels = np.zeros(5836)
train_gallery_camId = np.zeros(5836)

train_query = np.zeros((1532, 2048))
train_query_labels = np.zeros(1532)
train_query_camId = np.zeros(1532)

for i in range(7368):
    if camId_train[i] == 1 and camId_train[i+1] == 2:
        train_query[count_query, :] = features_train[i, :]
        train_query_labels[count_query] = labels_train[i]
        train_query_camId[count_query] = camId_train[i]
        count_query += 1
        
    elif camId_train[i] == 2 and camId_train[i-1] == 1:
        train_query[count_query, :] = features_train[i, :]
        train_query_labels[count_query] = labels_train[i]
        train_query_camId[count_query] = camId_train[i]
        count_query += 1
    
    else:
        train_gallery[count_gallery, :] = features_train[i, :]
        train_gallery_labels[count_gallery] = labels_train[i]
        train_gallery_camId[count_gallery] = camId_train[i]
        count_gallery += 1
        

#%% apply Kernel to train_gallery. If the kernel was applied my mistake than re-run previous section.

model_k= KernelPCA(kernel = "rbf", gamma = 0.0001)
model_k.fit(train_gallery)
train_gallery = model_k.transform(train_gallery)
train_query = model_k.transform(train_query)
        
#%% KNN on training with different distance functions, remove the "#" from the
# one you want to use, only one at the time. Then choose between the normal kNN
#and the improved kNN  algorithm uncommenting the one that you prefer.

#To implement the weighted improved kNN substitute this line "voted_label = np.bincount(majority_l)"
# in the improved kNN with "voted_label = np.bincount(majority_l, weights=w)"

get_min = np.zeros(len(train_gallery))
min_matrix = np.zeros(len(train_query))
accuracy_time = np.zeros((3,2))
count = 0
w = np.flip(range(1,21), axis=0)

for k in 1,5,10:
    majority_l = np.zeros(20)
    majority_k = np.zeros(k)
    accuracy_k = 0
    start = time.time() #start timer
    for i in range(len(train_query)): #loop thourgh the query images
        for j in range(len(train_gallery)): #comprare each image of query to the images in gallery
            if train_query_labels[i] == train_gallery_labels[j] and train_query_camId[i] == train_gallery_camId[j]: #Check if image has same label and camera
                get_min[j] = 10000000
            else:
                #get_min[j] = distance.euclidean(train_query[i,:], train_gallery[j,:]) #function used to compare query to gallery
                #get_min[j] = distance.cosine(train_query[i,:], train_gallery[j,:]) #cosine distance
                #get_min[j] = distance.cityblock(train_query[i,:], train_gallery[j,:]) #manhattan distance
                #get_min[j] = distance.chebyshev(train_query[i,:], train_gallery[j,:]) #Chebyshev distance
                #get_min[j] = distance.braycurtis(train_query[i,:], train_gallery[j,:]) #Bray-Curtis distance
                #get_min[j] = distance.minkowski(train_query[i,:], train_gallery[j,:], p =3) #Minkowski distance with p = 3
                #get_min[j] = distance.chebyshev(train_query[i,:], train_gallery[j,:], p =4) #Minkowski distance with p = 4
                #get_min[j] = distance.correlation(train_query[i,:], train_gallery[j,:]) #Correlation distance

                
                
#*******************normal kNN*******************
#        majority = np.argsort(get_min) #return vector with the indexes "get_min" vector in order
#        for l in range(k):
#            majority_k[l] = train_gallery_labels[int(majority[l])] #return a vector with k labels closest to the query image
#            if majority_k[l] == train_query_labels[i]:
#                accuracy_k += 1 #i.e. this means the correct label has appeared within the chosen rank and we have fulfilled our accuracy measure
#                break
                
        
#*******************improved kNN*******************
#        majority = np.argsort(get_min) #return vector with the indexes "get_min" vector in order
#        for l in range(20):
#            majority_l[l] = train_gallery_labels[int(majority[l])] #return a vector with k labels closest to the query image
#        majority_l = majority_l.astype(int) #needed to use np.bincount()
#        voted_label = np.bincount(majority_l) #return the most common label
#        voted_label = np.flip(np.argsort(voted_label), axis=0)
#        for m in range(k):
#            if voted_label[m] == train_query_labels[i]:
#                accuracy_k += 1 #i.e. this means the correct label has appeared within the chosen rank and we have fulfilled our accuracy measure
#                break
            
    end = time.time() #end timer
    accuracy_time[count,0] = (accuracy_k/len(train_query))*100
    accuracy_time[count,1] = end - start
    count += 1
    
print(accuracy_time) #first column is the accuracy for k=1,5,10 and the second columnis the computing time


#%% K-Means on training with Euclidean distance
    
kmeans = KMeans(n_clusters=766, random_state=0).fit(train_gallery)
cluster_data = kmeans.labels_ #(5328,) gives the label/centre that each sample belongs to 
centres = kmeans.cluster_centers_ #feature vector for each cluster centre
get_min = np.zeros(766)

accuracy_time = np.zeros(3)
count = 0

for k in 1,5,10:
    accuracy_k = 0
    majority_m = np.zeros(k)
    for i in range (1532):#number of query features
        nearest_cluster_points = []
    
        for j in range (766): #find closest cluster to the query vector
            get_min[j] = distance.euclidean(train_query[i,:],centres[j,:])
        min_clus = np.argsort(get_min)
        min_cluster = min_clus[0] #gives us cluster that the sample is associated with
        
        for s in range (5386): #group the vectors from the selected cluster into a matrx
            if cluster_data[s] == min_cluster:
                nearest_cluster_points.append(s)
                
        new_get_min = np.zeros(len(nearest_cluster_points))
        
        for m in range (len(nearest_cluster_points)): #KNN inside cluster
            new_get_min[m] = distance.euclidean(train_query[i,:],centres[j,:])
            
        new_min_clus = np.argsort(new_get_min) #indexes of new_get_min in order
        
        if k > len(nearest_cluster_points):
            k = len(nearest_cluster_points)
              
        for l in range (k):  
            s = nearest_cluster_points[int(new_min_clus[l])]          
            if train_query_labels[i] == train_gallery_labels[s] and train_query_camId[i] == train_gallery_camId[s]:
                pass
            elif train_query_labels[i] == train_gallery_labels[s]:
                accuracy_k+=1 #i.e. this means the correct label has appeared within the chosen rank and we have fulfilled our accuracy measure
                break
        
    accuracy_time[count] = (accuracy_k/len(train_query))*100
    count += 1
                

print(accuracy_time) #accuracy for k=1,5,10


#%% apply Kernel to the testing data. If the kernel was applied my mistake than re-run the fourth section.

model_k= KernelPCA(kernel = "rbf", gamma = 0.0001)
model_k.fit(features_gallery)
features_gallery = model_k.transform(features_gallery)
features_query = model_k.transform(features_query)

#%% KNN on testing with different distance formulas, remove the "#" from the 
# one you want to use, only one at the time. Then choose between the normal kNN
#and the improved kNN algorithm uncommenting the one that you prefer.

#To implement the weighted improved kNN substitute this line "voted_label = np.bincount(majority_l)"
# in the improved kNN with "voted_label = np.bincount(majority_l, weights=w)"

min_matrix = np.zeros(len(query_idx))
accuracy_time = np.zeros((3,2))
count = 0
w = np.flip(range(1,21), axis=0)

for k in 1,5,10: #1,5,10
    majority_l = np.zeros(20)
    majority_k = np.zeros(k)
    accuracy_k = 0
    start = time.time() #start timer
    for i in range(len(query_idx)): #loop thourgh the query images
        for j in range(len(gallery_idx)): #comprare each image of query to the images in gallery
            if labels_query[i] == labels_gallery[j] and camId_query[i] == camId_gallery[j]: #Check if image has same label and camera
                get_min[j]=10000000
            else:
                #get_min[j] = distance.euclidean(features_query[i,:], features_gallery[j,:]) #function used to compare query to gallery
                #get_min[j] = distance.cosine(features_query[i,:], features_gallery[j,:])
                #get_min[j] = distance.cityblock(features_query[i,:], features_gallery[j,:]) #manhattan distance
                #get_min[j] = distance.chebyshev(features_query[i,:], features_gallery[j,:]) #Chebyshev distance 
                #get_min[j] = distance.braycurtis(features_query[i,:], features_gallery[j,:]) #Bray-Curtis distance
                #get_min[j] = distance.minkowski(features_query[i,:], features_gallery[j,:], p =3) #Minkowski distance with p = 3
                #get_min[j] = distance.Minkowski(features_query[i,:], features_gallery[j,:], p =4) #Minkowski distance with p = 4
                #get_min[j] = distance.correlation(features_query[i,:], features_gallery[j,:]) #Correlation distance
    
             
#*******************normal kNN*******************
#            majority = np.argsort(get_min) #return vector with the indexes "get_min" vector in order
#            for l in range(k):
#                majority_k[l] = labels_gallery[int(majority[l])] #return a vector with k labels closest to the query image
#                if majority_k[l] == labels_query[i]:
#                    accuracy_k+=1 #i.e. this means the correct label has appeared within the chosen rank and we have fulfilled our accuracy measure
#                    break
                
                
#*******************improved kNN*******************
#        majority = np.argsort(get_min) #return vector with the indexes "get_min" vector in order
#        for l in range(20):
#            majority_l[l] = labels_gallery[int(majority[l])] #return a vector with k labels closest to the query image
#        majority_l = majority_l.astype(int) #needed to use np.bincount()
#        voted_label = np.bincount(majority_l) #return the most common label
#        voted_label = np.flip(np.argsort(voted_label), axis=0)
#        for m in range(k):
#            if voted_label[m] == labels_query[i]:
#                accuracy_k += 1 #i.e. this means the correct label has appeared within the chosen rank and we have fulfilled our accuracy measure
                    
                
    end = time.time() #end timer
    accuracy_time[count,0] = (accuracy_k/len(features_query))*100
    accuracy_time[count,1] = end-start
    count += 1
    
print(accuracy_time) #first column is the accuracy for k=1,5,10 and the second columnis the computing time


#%% K-Means on testing with Euclidean distance
    
kmeans = KMeans(n_clusters=700, random_state=0).fit(features_gallery)
cluster_data = kmeans.labels_ #(5328,) gives the label/centre that each sample belongs to 
centres = kmeans.cluster_centers_ #feature vector for each cluster centre
get_min = np.zeros(700)

accuracy_time = np.zeros(3)
count = 0

for k in 1,5,10:
    accuracy_k = 0
    majority_m = np.zeros(k)
    for i in range (1400):#number of query features
        nearest_cluster_points = []
    
        for j in range (700): #find closest cluster to the query vector
            get_min[j] = distance.euclidean(features_query[i,:],centres[j,:])
        min_clus = np.argsort(get_min)
        min_cluster = min_clus[0] #gives us cluster that the sample is associated with
        
        for s in range (5328): #group the vectors from the selected cluster into a matrx
            if cluster_data[s] == min_cluster:
                nearest_cluster_points.append(s)
                
        new_get_min = np.zeros(len(nearest_cluster_points))
        
        for m in range (len(nearest_cluster_points)): #KNN inside cluster
            new_get_min[m] = distance.euclidean(features_query[i,:],centres[j,:])
            
        new_min_clus = np.argsort(new_get_min) #indexes of new_get_min in order
        
        if k > len(nearest_cluster_points):
            k = len(nearest_cluster_points)
              
        for l in range (k):  
            s = nearest_cluster_points[int(new_min_clus[l])]          
            if labels_query[i] == labels_gallery[s] and camId_query[i] == camId_gallery[s]:
                pass
            elif labels_query[i] == labels_gallery[s]:
                accuracy_k+=1 #i.e. this means the correct label has appeared within the chosen rank and we have fulfilled our accuracy measure
                break
        
    accuracy_time[count] = (accuracy_k/len(features_query))*100
    count += 1
                

print(accuracy_time) #accuracy for k=1,5,10
