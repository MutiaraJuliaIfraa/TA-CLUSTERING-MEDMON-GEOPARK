# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 23:03:10 2023

@author: Ifra
"""

import math
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self, point):
        self.points = [point]
        self.centroid = point

    def merge(self, other_cluster):
        self.points.extend(other_cluster.points)
        self.update_centroid()

    def update_centroid(self):
        num_points = len(self.points)
        dim = len(self.centroid)

        new_centroid = [0] * dim
        for point in self.points:
            for i in range(dim):
                new_centroid[i] += point[i]

        self.centroid = [coord / num_points for coord in new_centroid]

def euclidean_distance(p1, p2):
    squared_sum = sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))
    return math.sqrt(squared_sum)

def complete_linkage_clustering(points, k):
    clusters = [Cluster(point) for point in points]

    while len(clusters) > k:
        min_distance = float('inf')
        merge_clusters = None

        for i in range(len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                max_distance = float('-inf')
                for p1 in clusters[i].points:
                    for p2 in clusters[j].points:
                        dist = euclidean_distance(p1, p2)
                        if dist > max_distance:
                            max_distance = dist

                if max_distance < min_distance:
                    min_distance = max_distance
                    merge_clusters = (i, j)

        if merge_clusters is not None:
            i, j = merge_clusters
            clusters[i].merge(clusters[j])
            del clusters[j]

    return clusters

# Membaca data dari file CSV
def read_csv(file_path, columns):
    df = pd.read_csv(file_path)
    points = df[columns].values.tolist()
    return points

# Menampilkan dendrogram
def plot_dendrogram(points):
    Z = linkage(points, method='complete', metric='euclidean')

    plt.figure(figsize=(10, 5))
    plt.title('Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')

    dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)

    plt.show()

def silhouette_score(points, clusters):
    silhouette_values = []

    for i, point in enumerate(points):
        a_i = 0
        b_i = float('inf')

        cluster_i = None
        for cluster in clusters:
            if point in cluster.points:
                cluster_i = cluster
                break

        for other_cluster in clusters:
            if other_cluster != cluster_i:
                sum_distance = 0
                count = 0
                for other_point in other_cluster.points:
                    distance = euclidean_distance(point, other_point)
                    sum_distance += distance
                    count += 1
                average_distance = sum_distance / count
                b_i = min(b_i, average_distance)

        sum_distance = 0
        count = 0
        for p in cluster_i.points:
            distance = euclidean_distance(point, p)
            sum_distance += distance
            count += 1
        a_i = sum_distance / (count - 1) if count > 1 else 0

        if a_i != 0 or b_i != 0:
            silhouette_value = (b_i - a_i) / max(a_i, b_i)
            silhouette_values.append(silhouette_value)

            # Menampilkan nilai a dan b
            #print(f"Point {i+1}: a_i = {a_i}, b_i = {b_i}")

    if silhouette_values:
        silhouette_avg = sum(silhouette_values) / len(silhouette_values)
        return silhouette_avg
    else:
        return 0
    
# Contoh penggunaan
file_path = 'raw.csv'  # Ubah dengan path file CSV yang sesuai
columns = ['bobot_isi_berita', 'bobot_tema_berita']
k = 3

points = read_csv(file_path, columns)
result = complete_linkage_clustering(points, k)

# Menampilkan hasil clustering
for i, cluster in enumerate(result):
    print(f'Cluster {i + 1}: {cluster.points}')
    
# Menampilkan dendrogram
plot_dendrogram(points)

# Menghitung Silhouette Score, cohesion, dan separation
silhouette_avg = silhouette_score(points, result)
print(f"Silhouette Score: {silhouette_avg}")

silhouette_sklearn = silhouette_score(points, result)
print(f'Silhouette Score dengan Library : {silhouette_sklearn}')

# Menghitung Silhouette Score untuk berbagai jumlah kluster
silhouette_scores = []
for k in range(2, 11):
    clusters = complete_linkage_clustering(points, k)
    flat_points = [point for cluster in clusters for point in cluster.points]
    labels = [idx for idx, cluster in enumerate(clusters) for _ in cluster.points]
    silhouette_avg = silhouette_score(points, clusters)
    silhouette_scores.append(silhouette_avg)

# Menampilkan grafik Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, 'b--', marker='o')
plt.title('Silhouette Score', fontsize=15)
plt.xlabel('Number of Clusters', fontsize=15)
plt.ylabel('Silhouette Score', fontsize=15)
plt.xticks(range(2, 11))
plt.show()