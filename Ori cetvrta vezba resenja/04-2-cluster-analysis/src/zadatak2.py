from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from kmeans import KMeans2
from dbscan import DBScan
import numpy as np
import pandas as pd

# --- UCITAVANJE I PRIKAZ IRIS DATA SETA --- #

dataset  = pd.read_csv('../data/dataset.csv')
x = dataset[['book1', 'book2', 'book3', 'book4', 'book5']].sum(axis=1)
sum_column = dataset["book1"] + dataset["book2"]+dataset["book3"] + dataset["book4"]+dataset["book5"]
dataset["suma"] = sum_column
iris_data = x
print(dataset["suma"])
print(len(dataset["suma"]))
print(dataset["suma"][1])
print(dataset["suma"][2])
print(dataset["suma"][3])

iris_data =[]
plt.figure()
for i in range(len(dataset["suma"])):
    plt.scatter([i, 0], [dataset["suma"][i], 1])
    iris_data.append((i, dataset["suma"][i]))
plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.show()


# --- INICIJALIZACIJA I PRIMENA K-MEANS ALGORITMA --- #

# TODO 2: K-means na Iris data setu
kmeans = KMeans2(n_clusters=3, max_iter=100)
kmeans.fit(iris_data,True)

colors = {0: 'red', 1: 'green',2:'blue',3:'yellow'}
plt.figure()
for idx, cluster in enumerate(kmeans.clusters):
    plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
    for datum in cluster.data:  # iscrtavanje tacaka
        plt.scatter(datum[0], datum[1], c=colors[idx])

plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.show()


# # --- ODREDJIVANJE OPTIMALNOG K --- #
#
# plt.figure()
# sum_squared_errors = []
# for n_clusters in range(2, 10):
#     kmeans = KMeans2(n_clusters=n_clusters, max_iter=7)
#     kmeans.fit(iris_data,True)
#     sse = kmeans.sum_squared_error()
#     sum_squared_errors.append(sse)
# print( sum_squared_errors)
#
# plt.style.use("fivethirtyeight")
# plt.xticks(range(0, len(sum_squared_errors)))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.plot(range(0,  len(sum_squared_errors)), sum_squared_errors)
# plt.show()


# # TODO 7: DBSCAN nad Iris podacima, prikazati rezultate na grafiku isto kao kod K-means
print("DBSCAN")
dbscan = DBScan(epsilon=1.2,min_points=3)
dbscan.fit(iris_data)

colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'pink', 4: 'red', 5: 'grey', 6: 'yellow',
          7: 'red', 8: 'white', 9: 'brown', 10: 'pink', 11: 'orange', 12: 'black', 13: 'darkblue'}
plt.figure()

redni = 0
for idx, cluster in enumerate(dbscan.clusters):
    for datum in cluster.data:  # iscrtavanje tacaka
        plt.scatter(datum[0], datum[1], c=colors[idx])

for datum in cluster.data:  # iscrtavanje tacaka
    plt.scatter(datum[0], datum[1], c=colors[idx])

plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.show()


