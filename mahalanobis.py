from scipy.io import loadmat
import numpy as np
import json
from collections import Counter
from scipy.spatial import distance

all_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')

camId = all_idxs['camId'].flatten()
filelist = all_idxs['filelist'].flatten()
labels = all_idxs['labels'].flatten()

train_idx = all_idxs['train_idx'].flatten() - 1 #subtract one to go from Matlab indexing to Python indexing
gallery_idx = all_idxs['gallery_idx'].flatten() - 1
query_idx = all_idxs['query_idx'].flatten() - 1

#%% load features and convert it to a np array
with open('feature_data.json', 'r') as f: 
 features = json.load(f)

features = np.asarray(features)

#%% split the feature matrix in training, gallery and query matrices

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
    
#%% training
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
        
#%%Mahalanobis based on algorithm in the following paperhttps://www.sciencedirect.com/science/article/pii/S0031320308002057
#Conducting on training set
d=2048 #low-dimensional parameter
e=10 #error parameter

sim_pairs=np.zeros((2048,2048))
diff_pairs=np.zeros((2048,2048))
for i in range (5):
    sim=np.zeros((2048,2048))
    diff=np.zeros((2048,2048))
    for j in range (5):
        if train_query_labels[i]==train_gallery_labels[j]:
            sim=((train_query[i,:]-train_gallery[j,:]).T)*((train_query[i,:]-train_gallery[j,:]))
            sim_pairs=sim_pairs+sim #covariance of similar points
        if train_query_labels[i]!=train_gallery_labels[j]:   
            diff=((train_query[i,:]-train_gallery[j,:]).T)*((train_query[i,:]-train_gallery[j,:]))
            diff_pairs=diff_pairs+diff #covariance of dissimilar points

r=np.linalg.matrix_rank(sim_pairs) #calculate rank of covariance matrix of similar points
n=2048 #number of features

eigvals_sim, eigvecs_sim = np.linalg.eigh(sim_pairs) 
eigvals_diff, eigvecs_diff = np.linalg.eigh(diff_pairs) 
    
eigvals_diff = np.flip(eigvals_diff, 0) #this orders eigenvalues with lowest first 

alpha=0
beta=0
if d>n-r:
    Sw=np.matrix.trace(sim_pairs)
    Sb=np.matrix.trace(diff_pairs)
    lambda_1 = Sb/Sw
    
    for i in range (d):
        alpha=alpha+eigvals_diff[i]
        beta=beta+eigvals_sim[i]
    lambda_2=alpha/beta
    
    lambda_final=(lambda_1+lambda_2)/2
    
    while lambda_2 - lambda_1 > e:
        eigvals, eigvecs= np.linalg.eigh(diff_pairs-lambda_final*sim_pairs) 
        eigvals = np.flip(eigvals, 0) #so largest is first
        for i in range (d):
            g=g+eigvals[i]
        
        if g>0:
            lambda_1=lambda_final
        else:
            lambda_2=lambda_final
        lambda_final=(lambda_1 + lambda_2)/2 #iteratively edit value of lambda_final
    final_eigvals, final_eigvecs= np.linalg.eigh(diff_pairs-(lambda_final*sim_pairs)) 
    final_eigvals = np.flip(final_eigvals, 0) 
    final_eigvecs = np.flip(final_eigvecs, 0) 
    
    W=final_eigvecs[:,:d]
    
if d<n-r:
    Z=eigvecs_sim[:,:n]
    print(Z.shape)
    
    matrix=np.matmul(Z.T, diff_pairs)
    
    eigvals_z,eigvecs_z=np.linalg.eigh(np.matmul(matrix,Z))
    eigvals_z = np.flip(eigvals_z, 0) 
    eigvecs_z = np.flip(eigvecs_z, 0) 
    
    W=eigvecs_z[:,:d]

A=np.matmul(W,W.T) #learned Mahalanobis distance metric

G=W.T 

features_query_new=np.matmul(features_query,G.T)
features_gallery_new=np.matmul(features_gallery,G.T) #transform data set with learned distance metric
#%% kNN 
get_min = np.zeros(5836)
min_matrix = np.zeros(1532)
accuracy = 0
counter = 0
k = 1 #number of candidates for majority voting
majority_k = np.zeros(k)
accuracy_k = 0

for i in range(1532): #loop thourgh the query images
    for j in range(5836): #comprare each image of query to the images in gallery
        if train_query_labels[i] ==train_gallery_labels[j] and train_query_camId[i] == train_gallery_camId[j]: #Check if image has same label and camera
            counter += 1
            get_min[j]=10000000
        else:
            get_min[j] = distance.euclidean((train_query[i,:]),(train_gallery[j,:])) #function used to compare query to gallery
            
    #majority voting with k numbers of candidates
    majority = np.argsort(get_min) #return vector with the indexes "get_min" vector in order
    for l in range(k):
        majority_k[l] = train_gallery_labels[int(majority[l])] #return a vector with k labels closest to the query image
        if majority_k[l]==train_query_labels[i]:
            accuracy_k+=1 #i.e. this means the correct label has appeared within the chosen rank and we have fulfilled our accuracy measure
            break
        #print(l) #testing whether break breaks whole for loop rather than just if statement - it does

percent_accuracy=(accuracy_k/1532)*100

print(counter)
print(accuracy_k)
print(percent_accuracy)

#%%Conducting Mahalanobis on testing set
d=2048 #low-dimensional parameter
e=10 #error parameter

sim_pairs=np.zeros((2048,2048))
diff_pairs=np.zeros((2048,2048))
for i in range (5):
    sim=np.zeros((2048,2048))
    diff=np.zeros((2048,2048))
    for j in range (5):
        if labels_query[i]==labels_gallery[j]:
            sim=((features_query[i,:]-features_gallery[j,:]).T)*((features_query[i,:]-features_gallery[j,:]))
            sim_pairs=sim_pairs+sim #covariance of similar points
        if labels_query[i]!=labels_gallery[j]:   
            diff=((features_query[i,:]-features_gallery[j,:]).T)*((features_query[i,:]-features_gallery[j,:]))
            diff_pairs=diff_pairs+diff #covariance of dissimilar points

r=np.linalg.matrix_rank(sim_pairs) #calculate rank of covariance matrix of similar points
n=2048 #number of features

eigvals_sim, eigvecs_sim = np.linalg.eigh(sim_pairs) 
eigvals_diff, eigvecs_diff = np.linalg.eigh(diff_pairs) 
    
eigvals_diff = np.flip(eigvals_diff, 0) #this orders eigenvalues with lowest first 

alpha=0
beta=0
if d>n-r:
    Sw=np.matrix.trace(sim_pairs)
    Sb=np.matrix.trace(diff_pairs)
    lambda_1 = Sb/Sw
    
    for i in range (d):
        alpha=alpha+eigvals_diff[i]
        beta=beta+eigvals_sim[i]
    lambda_2=alpha/beta
    
    lambda_final=(lambda_1+lambda_2)/2
    
    while lambda_2 - lambda_1 > e:
        eigvals, eigvecs= np.linalg.eigh(diff_pairs-lambda_final*sim_pairs) 
        eigvals = np.flip(eigvals, 0) #so largest is first
        for i in range (d):
            g=g+eigvals[i]
        
        if g>0:
            lambda_1=lambda_final
        else:
            lambda_2=lambda_final
        lambda_final=(lambda_1 + lambda_2)/2 #iteratively edit value of lambda_final
    final_eigvals, final_eigvecs= np.linalg.eigh(diff_pairs-(lambda_final*sim_pairs)) 
    final_eigvals = np.flip(final_eigvals, 0) 
    final_eigvecs = np.flip(final_eigvecs, 0) 
    
    W=final_eigvecs[:,:d]
    
if d<n-r:
    Z=eigvecs_sim[:,:n]
    print(Z.shape)
    
    matrix=np.matmul(Z.T, diff_pairs)
    
    eigvals_z,eigvecs_z=np.linalg.eigh(np.matmul(matrix,Z))
    eigvals_z = np.flip(eigvals_z, 0) 
    eigvecs_z = np.flip(eigvecs_z, 0) 
    
    W=eigvecs_z[:,:d]

A=np.matmul(W,W.T) #learned Mahalanobis distance metric

G=W.T 

features_query_new=np.matmul(features_query,G.T)
features_gallery_new=np.matmul(features_gallery,G.T) #transform data set with learned distance metric
#%% kNN 
get_min = np.zeros(5328)
min_matrix = np.zeros(1400)
accuracy = 0
counter = 0
k = 1 #number of candidates for majority voting
majority_k = np.zeros(k)
accuracy_k = 0

for i in range(1400): #loop thourgh the query images
    for j in range(5328): #comprare each image of query to the images in gallery
        if labels_query[i] ==labels_gallery[j] and camId_query[i] == camId_gallery[j]: #Check if image has same label and camera
            counter += 1
            get_min[j]=10000000
        else:
            get_min[j] = distance.euclidean((features_query[i,:]),(features_gallery[j,:])) #function used to compare query to gallery
            
    #majority voting with k numbers of candidates
    majority = np.argsort(get_min) #return vector with the indexes "get_min" vector in order
    for l in range(k):
        majority_k[l] = labels_gallery[int(majority[l])] #return a vector with k labels closest to the query image
        if majority_k[l]==labels_query[i]:
            accuracy_k+=1 #i.e. this means the correct label has appeared within the chosen rank and we have fulfilled our accuracy measure
            break
        #print(l) #testing whether break breaks whole for loop rather than just if statement - it does

percent_accuracy=(accuracy_k/1400)*100

print(counter)
print(accuracy_k)
print(percent_accuracy)

#%%Mahalanobis implemented with imposter set, here done with training set
sim_pairs=np.zeros((2048,2048))
diff_pairs=np.zeros((2048,2048))

d_ij=np.zeros((1532,5836))
idx_d=[]
for i in range (1532):
    for j in range (5836):
        d_ij[i,j]=(np.sum(((train_query[i,:] - train_gallery[j,:]))**2)) #contain all distances between pairs
        if train_query_labels[i]!=train_gallery_labels[j]:
            idx_d.append([i,j]) #stores index within d_ij of dissimilar points
            
print(idx_d)

for i in range (5):
    sim=np.zeros((2048,2048))
    diff=np.zeros((2048,2048))
    
    for j in range (5):
        if train_query_labels[i]==train_gallery_labels[j]:
            sim=((train_query[i,:]-train_gallery[j,:]).T)*((train_query[i,:]-train_gallery[j,:]))
            sim_pairs=sim_pairs+sim #covariance of similar points
            
            for s in range(len(idx_d)):
                x=idx_d[s]
                first=x[0]
                second=x[1] 
                if d_ij[first,second]<=d_ij[i,j]:#we have identified an imposter invading perimeter of the similar pair
                    w_il=np.exp(-d_ij[first,second]/d_ij[i,j])
                    diff=(w_il*(train_query[i,:]-train_gallery[j,:]).T)*((train_query[i,:]-train_gallery[j,:]))
                    diff_pairs=diff_pairs+diff #covariance of dissimilar points

eigvals, eigvecs = np.linalg.eigh(sim_pairs-diff_pairs) 

#%%kNN
get_min = np.zeros(5836)
min_matrix = np.zeros(1532)
accuracy = 0
counter = 0
k = 10 #number of candidates for majority voting
majority_k = np.zeros(k)
accuracy_k = 0

train_query=np.matmul(train_query,eigvecs.T)
train_gallery=np.matmul(train_gallery,eigvecs.T)

for i in range(1532): #loop thourgh the query images
    for j in range(5836): #comprare each image of query to the images in gallery
        if train_query_labels[i] == train_gallery_labels[j] and train_query_camId[i] == train_gallery_camId[j]: #Check if image has same label and camera
            counter += 1
            get_min[j]=10000000
        else:
            get_min[j] = distance.euclidean((train_query[i,:]),(train_gallery[j,:])) #function used to compare query to gallery
            
    #majority voting with k numbers of candidates
    majority = np.argsort(get_min) #return vector with the indexes "get_min" vector in order
    for l in range(k):
        majority_k[l] = train_gallery_labels[int(majority[l])] #return a vector with k labels closest to the query image
        if majority_k[l]==train_query_labels[i]:
            accuracy_k+=1 #i.e. this means the correct label has appeared within the chosen rank and we have fulfilled our accuracy measure
            break
        #print(l) #testing whether break breaks whole for loop rather than just if statement - it does

percent_accuracy=(accuracy_k/1532)*100

print(counter)
print(accuracy_k)
print(percent_accuracy)



