# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:11:27 2023

@author: Ifra
"""

import numpy as np 
import pandas as pd 
from pandas import DataFrame
import warnings 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

#Membaca Data 
df = pd.read_csv('raw.csv')
df = DataFrame(df,columns=['bobot_isi_berita','bobot_tema_berita'])
print(f'Jumlah Baris : {df.shape[0]}')
print(df)

#Menampilkan Plot Menatahan Data
plt.figure(figsize=(6,6))
plt.scatter(df['bobot_isi_berita'], df['bobot_tema_berita'])
plt.xlabel('bobot_isi_berita', fontsize=15) 
plt.ylabel('bobot_tema_berita', fontsize=15)
plt.show()

X = df[['bobot_isi_berita', 'bobot_tema_berita']]

#Range Cluster 
inertia = []
for k in range(1,11): 
    kmean = KMeans(n_clusters=k)
    kmean.fit(X)
    iner = kmean.inertia_ 
    inertia.append(iner)

plt.figure(figsize=(16,5))
plt.plot(range(1,11), inertia, 'b--', marker='o')
plt.xlabel('n_clusters', fontsize=15)
plt.show()

X = df[['bobot_isi_berita', 'bobot_tema_berita']]
X_ori = X.copy()

#Run Model KMeans
model = KMeans(n_clusters=3)
model.fit(X)
centroids = model.cluster_centers_
label = model.predict(X)
X['label'] = label
print(X)
print(centroids)

my_color = sns.color_palette(['#ff0400','#210891','#54bf22'])
plt.figure(figsize=(7,7))
sns.scatterplot(data=X, x='bobot_isi_berita', y='bobot_tema_berita',c=kmean.labels_.astype(float), s=60, alpha=0.5, hue='label', palette=my_color)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=60)
plt.xlabel('bobot_isi_berita', fontsize=15) 
plt.ylabel('bobot_tema_berita', fontsize=15)
plt.title('Visualization of raw data');
plt.show()

#Menghitung jarak dengan ecuslidean distance
def euclidean(arr1, arr2): 
  arr1 = np.array(arr1)
  arr2 = np.array(arr2) 

  distance = np.linalg.norm(arr1 - arr2)
  return distance 

#Membuat Kolom Tuple dari bobot dan isi berita
X['distance'] = X[['bobot_isi_berita', 'bobot_tema_berita']].apply(tuple, axis=1)
print(X)  

#############################
silhouette_list = []
cohesion_list = []
separation_list = []
for i in range(0, len(X)): 
  random_p = X.loc[i] 
  X['distance from p'] = X.apply(lambda x: euclidean(x['distance'], random_p['distance']), axis=1)#Mengukur jarak antara random_p dengan seluruh data dalam tabel

  #Membuat Jarak Cohesion Manual 
  ai_data = X[X['label'] == random_p['label']]
  cohesion = ai_data['distance from p'].mean()
  cohesion_list.append(cohesion)
  
  #Membuat Jarak Separation Manual
  bi_data = X[X['label'] != random_p['label']]

  separation_list = []
  for i in bi_data['label'].unique(): #Looping untuk menghitung rata2 setiap label yg berbeda pada random_p
    sep = X[X['label'] == i] 
    separation = sep['distance from p'].mean()
    separation_list.append(separation)
  
  #Menghitung Silhouette Score Manual
  ai = cohesion 
  bi = min(separation_list)
  silhouette = (bi - ai)/max(ai, bi)
  silhouette_list.append(silhouette)
  
silhouette_from_all_sample = np.mean(silhouette_list)
#cohesion_mean = np.mean(cohesion_list)
#separation_mean = np.mean(separation_list)
#print(f"Cohesion Mean: {cohesion_mean}")
#print(f"Separation Mean: {separation_mean}")
print(f'Silhouette Score(n=3) : {silhouette_from_all_sample}')

silhouette_sklearn = silhouette_score(X_ori, label)
print(f'Silhouette Score(n=3) dengan Library : {silhouette_score(X_ori, label)}')

#Menghitung Silhouette Score Untuk Seluruh Cluster
silhouette_for_all_n_cluster = []
for n in range(2, 11): 
  X = df[['bobot_isi_berita', 'bobot_tema_berita']]
  model = KMeans(n_clusters=n) 
  model.fit(X)
  label = model.predict(X)
  X['label'] = label 

  X['distance'] = X[['bobot_isi_berita', 'bobot_tema_berita']].apply(tuple, axis=1)

  silhouette_list = []
  for i in range(0, len(X)): 
    random_p = X.loc[i] 
    X['distance from p'] = X.apply(lambda x: euclidean(x['distance'], random_p['distance']), axis=1) 

    #Membuat Jarak Cohesion Manual 
    ai_data = X[X['label'] == random_p['label']]
    cohesion = ai_data['distance from p'].mean()

    #Membuat Jarak Separation Manual 
    bi_data = X[X['label'] != random_p['label']]

    separation_list = []
    for i in bi_data['label'].unique(): 
      sep = X[X['label'] == i] 
      separation = sep['distance from p'].mean()
      separation_list.append(separation)

    #Menghitung Silhouette Score Manual
    ai = cohesion 
    bi = min(separation_list)
    silhouette = (bi - ai)/max(ai, bi)
    silhouette_list.append(silhouette)
  
  silhouette_mean_from_all_sample = np.mean(silhouette_list)
  silhouette_for_all_n_cluster.append(silhouette_mean_from_all_sample)

plt.figure(figsize=(10, 5))  
plt.plot(range(2, 11), silhouette_for_all_n_cluster, 'b--', marker='o', label='manual')
plt.xlabel('Number of Clusters', fontsize=15)
plt.ylabel('Silhouette Score', fontsize=15)
plt.title('Grafik Silhouette Score', fontsize=15)
plt.show()