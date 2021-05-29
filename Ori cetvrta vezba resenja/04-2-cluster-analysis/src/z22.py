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
figure, axis = plt.subplots(2, 2)
iris_data =[]
plt.figure()
for i in range(len(dataset["suma"])):
    if dataset["suma"][i] >= 3:
        plt.scatter([i, 0], [dataset["suma"][i], 1])
        axis[0, 0].scatter([i, 0], [dataset["suma"][i], 1])
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

broj1 = 0
broj2 = 0
broj3 = 0

procenat1 = 0
procenat2 = 0
procenat3 = 0


for idx, cluster in enumerate(kmeans.clusters):
    plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
    for datum in cluster.data:
        if idx == 0:
            broj1 += 1
            if datum[1] == 5:
                procenat1 += 1
        elif idx == 1:
            broj2 += 1
            if datum[1] == 5:
                procenat2 += 1
                print("tacka")
                print(datum[0])
        elif idx == 2:
            broj3 += 1
            if datum[1] == 5:
                procenat3 += 1


    for datum in cluster.data:  # iscrtavanje tacaka
        plt.scatter(datum[0], datum[1], c=colors[idx])
        axis[0, 1].scatter(datum[0], datum[1], c=colors[idx])
plt.xlabel('suma')
plt.ylabel('S.NO')

print( "broj tacki u klasteru  " +  colors[0]  +" sa preko 5")
print( procenat1)
print( "broj tacki u klasteru  " +  colors[1]  +" sa preko 5")
print(procenat2)
print( "broj tacki u klasteru  " +  colors[2]  +" sa preko 5")
print(procenat3)

print( "broj tacki u klasteru  " +  colors[0] )
print(broj1)
print( "broj tacki u klasteru  " +  colors[1] )
print(broj2)
print( "broj tacki u klasteru  " +  colors[2] )
print(broj3)

procenat1 = procenat1 / broj1
procenat2 = procenat2 / broj2
procenat3 = procenat3 / broj3
print( "procenat   " + colors[0]+"s")
print( procenat1)
print( "procenat   " + colors[1]+"s")
print(procenat2)
print( "procenat   " + colors[2]+"s")
print(procenat3)

#plt.show()


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
#plt.plot(range(0,  len(sum_squared_errors)), sum_squared_errors)
axis[1, 0].plot(range(0,  len(sum_squared_errors)), sum_squared_errors)
plt.show()


# # TODO 7: DBSCAN nad Iris podacima, prikazati rezultate na grafiku isto kao kod K-means
# print("DBSCAN")
# print(iris_data)
# dbscan = DBScan(epsilon=10.2,min_points=3)
# dbscan.fit(iris_data)
#
# colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'pink', 4: 'red', 5: 'grey', 6: 'yellow',
#           7: 'red', 8: 'white', 9: 'brown', 10: 'pink', 11: 'orange', 12: 'black', 13: 'darkblue',14: 'red',
#           15: 'red', 16: 'white', 17: 'brown', 18: 'pink', 19: 'orange', 20: 'black', 21: 'darkblue',22: 'red',
#           23: 'red', 24: 'white', 25: 'brown', 26: 'pink', 27: 'orange', 28: 'black', 29: 'darkblue', 30: 'red',
#           31: 'red', 32: 'white', 33: 'brown', 34: 'pink', 35: 'orange', 36: 'black', 37: 'darkblue', 38: 'red'
#           }
# plt.figure()
#
# redni = 0
# print(dbscan.clusters)
# for idx, cluster in enumerate(dbscan.clusters):
#     print(cluster.data[0])
#     for datum in cluster.data:  # iscrtavanje tacaka
#         plt.scatter(datum[0], datum[1], c=colors[idx])
#         print(idx)
# plt.xlabel('Sepal width')
# plt.ylabel('Petal length')
# plt.show()