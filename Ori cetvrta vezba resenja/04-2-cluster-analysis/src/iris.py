from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from kmeans import KMeans2
from dbscan import DBScan


# --- UCITAVANJE I PRIKAZ IRIS DATA SETA --- #

iris_data = load_iris()  # ucitavanje Iris data seta
print(iris_data)
iris_data = iris_data.data[:, 1:3]  # uzima se druga i treca osobina iz data seta (sirina sepala i duzina petala)

plt.figure()
for i in range(len(iris_data)):
    plt.scatter(iris_data[i, 0], iris_data[i, 1])

plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.show()


# --- INICIJALIZACIJA I PRIMENA K-MEANS ALGORITMA --- #

# TODO 2: K-means na Iris data setu
kmeans = KMeans2(n_clusters=2, max_iter=100)
kmeans.fit(iris_data,True)

colors = {0: 'red', 1: 'green'}
plt.figure()
for idx, cluster in enumerate(kmeans.clusters):
    plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
    for datum in cluster.data:  # iscrtavanje tacaka
        plt.scatter(datum[0], datum[1], c=colors[idx])

plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.show()


# --- ODREDJIVANJE OPTIMALNOG K --- #

plt.figure()
sum_squared_errors = []
for n_clusters in range(2, 10):
    kmeans = KMeans2(n_clusters=n_clusters, max_iter=7)
    kmeans.fit(iris_data,True)
    sse = kmeans.sum_squared_error()
    sum_squared_errors.append(sse)
print( sum_squared_errors)

plt.style.use("fivethirtyeight")
plt.xticks(range(0, len(sum_squared_errors)))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.plot(range(0,  len(sum_squared_errors)), sum_squared_errors)
plt.show()


# TODO 7: DBSCAN nad Iris podacima, prikazati rezultate na grafiku isto kao kod K-means
dbscan = DBScan(epsilon=0.5, min_points=3)
dbscan.fit(iris_data)

colors = {0: 'red', 1: 'green', 2: 'blue'}
plt.figure()

for idx, cluster in enumerate(dbscan.clusters):
    for datum in cluster.data:  # iscrtavanje tacaka
        plt.scatter(datum[0], datum[1], c=colors[idx])

plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.show()