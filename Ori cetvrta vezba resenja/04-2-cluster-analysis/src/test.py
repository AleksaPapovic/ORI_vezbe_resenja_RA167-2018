import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics as sm
import math
from kmeans import KMeans2

from sklearn.datasets import load_iris

iris = load_iris()
print (iris.data)

print (iris.target_names)

x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width',
                                     'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

x.head()

iris_data = load_iris()  # ucitavanje Iris data seta
print(iris_data)
iris_data = iris_data.data[:, 1:3]  # uzima se druga i treca osobina iz data seta (sirina sepala i duzina petala)

plt.figure()
for i in range(len(iris_data)):
    plt.scatter(iris_data[i, 0], iris_data[i, 1])


plt.show()

iris_k_mean_model = KMeans(n_clusters=3)
iris_k_mean_model.fit(x)


kmeans2 = KMeans2(n_clusters=2, max_iter=100)
kmeans2.fit(iris_data)

colors = {0: 'red', 1: 'green'}
plt.figure()
for idx, cluster in enumerate(kmeans2.clusters):
    plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
    print(cluster.data)
    for datum in cluster.data:  # iscrtavanje tacaka
        plt.scatter(datum[0], datum[1], c=colors[idx])
plt.show()

for k, col in zip(range(iris_k_mean_model.n_clusters), colors):
    cluster_center = iris_k_mean_model.cluster_centers_[k]
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor='r',
            markeredgecolor='k', markersize=6)


plt.show()


dist_points_from_cluster_center = []
K = range(1,10)
for no_of_clusters in K:
  k_model = KMeans(n_clusters=no_of_clusters)
  k_model.fit(x)
  dist_points_from_cluster_center.append(k_model.inertia_)
  print(k_model.inertia_)


plt.plot(K, dist_points_from_cluster_center)
plt.show()