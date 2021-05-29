import numpy
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from kmeans import KMeans2
from dbscan import DBScan
import numpy as np
import pandas as pd
from tqdm import trange, tqdm

# --- UCITAVANJE I PRIKAZ IRIS DATA SETA --- #

dataset  = pd.read_csv('../data/dataset1.csv')
iris_data = dataset.iloc[: , [11,1,2,3,8,9]].values

for i in trange(len(iris_data)):
        if iris_data[i,1] == 'Male':
                iris_data[i,1] = 1
        elif iris_data[i,1] == 'Female':
                iris_data[i, 1] = 3
        else:
                iris_data[i, 1] = 2

iris_data = iris_data[:,[0,1,2,3,4,5]]

print(iris_data)

dataframe = pd.DataFrame(iris_data, columns = ['stroke','gender','age','hypertension','avg_glucose_level','bmi'])
dataframe.bmi.fillna(0, inplace=True)


print(dataframe)
# plt.figure()
# for i in trange(1):
#         plt.scatter(iris_data[i, 0], iris_data[i, 1])
#         plt.scatter(iris_data[i, 0], iris_data[i, 2])
#         plt.scatter(iris_data[i, 0], iris_data[i, 3])
#         print(iris_data[i,1])
#         print(iris_data[i,2])
#         print(iris_data[i,3])
# plt.xlabel('Sepal width')
# plt.ylabel('Petal length')
# plt.show()

broj1 = 0
broj2 = 0
broj3 = 0

procenat1_0 = 0
procenat2_0 = 0
procenat3_0 = 0
procenat1_1 = 0
procenat2_1 = 0
procenat3_1 = 0

# --- INICIJALIZACIJA I PRIMENA K-MEANS ALGORITMA --- #

# TODO 2: K-means na Iris data setu
kmeans = KMeans2(n_clusters=3, max_iter=100)
kmeans.fit(numpy.array(dataframe))

colors = {0: 'red', 1: 'green',2:'blue',3:'yellow'}


def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


iteracija = 0
for idx, cluster in tqdm_enumerate(kmeans.clusters):
    plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
    for datum in cluster.data:  # iscrtavanje tacaka
        if idx == 0:
            broj1 += 1
            print(datum[0])
            if datum[0] == 0:
                procenat1_0 += 1
            elif datum[0] == 1:
                procenat1_1 += 1
        elif idx == 1:
            broj2 += 1
            if datum[0] == 0:
                procenat2_0 += 1
            elif datum[0] == 1:
                procenat2_1 += 1
        elif idx == 2:
            broj3 += 1
            if datum[0] == 0:
                procenat3_0 += 1
            elif datum[0] == 1:
                procenat3_1 += 1

print( "broj tacki u klasteru  " +  colors[0]  +" sa preko 1")
print( procenat1_1)
print( "broj tacki u klasteru  " +  colors[0]  +" sa preko 0")
print( procenat1_0)

print( "broj tacki u klasteru  " +  colors[1]  +" sa preko 1")
print( procenat2_1)
print( "broj tacki u klasteru  " +  colors[1]  +" sa preko 0")
print( procenat2_0)

print( "broj tacki u klasteru  " +  colors[2]  +" sa preko 1")
print( procenat2_1)
print( "broj tacki u klasteru  " +  colors[2]  +" sa preko 0")
print( procenat2_0)

print( "broj tacki u klasteru  " +  colors[0] )
print(broj1)
print( "broj tacki u klasteru  " +  colors[1] )
print(broj2)
print( "broj tacki u klasteru  " +  colors[2] )
print(broj3)

procenat1_0 = procenat1_0 / broj1
procenat2_0 = procenat2_0 / broj2
procenat3_0 = procenat3_0 / broj3
procenat1_1 = procenat1_1 / broj1
procenat2_1 = procenat2_1 / broj2
procenat3_1 = procenat3_1 / broj3

print( "procenat u  " + colors[0]+"s sa 0")
print( procenat1_0)
print( "procenat u  " + colors[0]+"s sa 1")
print( procenat1_1)
print( "procenat u  " + colors[1]+"s sa 0")
print( procenat2_0)
print( "procenat u  " + colors[1]+"s sa 1")
print( procenat2_1)
print( "procenat u  " + colors[2]+"s sa 0")
print( procenat3_0)
print( "procenat u  " + colors[2]+"s sa 1")
print( procenat3_1)



#       axis[0, 1].scatter(datum[0], datum[1], c=colors[idx])
# plt.xlabel('suma')
# plt.ylabel('S.NO')
#
#
#
# --- ODREDJIVANJE OPTIMALNOG K --- #

plt.figure()
sum_squared_errors = []
for n_clusters in range(2, 10):
    kmeans = KMeans2(n_clusters=n_clusters, max_iter=7)
    kmeans.fit(numpy.array(dataframe),True)
    sse = kmeans.sum_squared_error()
    sum_squared_errors.append(sse)
print( sum_squared_errors)

plt.style.use("fivethirtyeight")
plt.xticks(range(0, len(sum_squared_errors)))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.plot(range(0,  len(sum_squared_errors)), sum_squared_errors)
plt.show()
