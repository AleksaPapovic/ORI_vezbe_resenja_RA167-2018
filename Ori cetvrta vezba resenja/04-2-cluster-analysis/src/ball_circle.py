import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans2
# from dbscan import DBScan

np.random.seed(1337)

n = 10

s1 = np.ndarray(shape=(n, 2))
s2 = np.ndarray(shape=(n, 4))

data = []  # ovde se nalaze podaci, u vidu liste tacaka sa (x,y) koordinatama

plt.figure()
a = 0
b = 0
for i in range(n):
    a += n
    x1, y1 = 1, a
    s1[i] = (x1, y1)

    r2, theta2 = np.random.normal(5, 0.25), np.random.uniform(0, 2*np.pi)
    b += n
    x2, y2 = 2, b
    s2[i] = (x2, y2, r2, theta2)

    plt.scatter(x1, y1)
    plt.scatter(x2, y2)

    data.append((x1, y1))
    data.append((x2, y2))

plt.show()

# TODO 5: K-means nad ovim podacima

# kmeans = KMeans2(n_clusters=2,max_iter=100)
# kmeans.fit(data ,True)
#
# colors = {0: 'red', 1: 'green'}
# plt.figure()
# for idx, cluster in enumerate(kmeans.clusters):
#     plt.scatter(cluster.center[0], cluster.center[1], c =colors[idx], marker= 'x', s=200)
#     for datum in cluster.data:
#         plt.scatter(datum[0],datum[1], c= colors[idx])
#
# plt.show()

# TODO 7: DBSCAN nad ovim podacima
#dbscan = DBScan(epsilon=1.2,min_points=3)
#dbscan.fit(data)

from sklearn.cluster import DBSCAN

db_default = DBSCAN(eps = 22, min_samples = 2).fit(data)

colors = {0: 'red', 1: 'pink',2: 'yellow', 3: 'cyan',4: 'green', 5: 'blue',6: 'black',7: 'grey',8: 'brown',9: 'orange'}
# plt.figure()

# for idx, cluster in enumerate(dbscan.clusters):
#     for datum in cluster.data:  # iscrtavanje tacaka
#         plt.scatter(datum[0], datum[1], c=colors[idx%6])
#         print(idx)
# plt.show()



# Building the label to colour mapping
colours = {}
colours[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'



plt.figure()
# For the construction of the legend of the plot
r = plt.scatter(data[0], data[1], color='r');
g = plt.scatter(data[0], data[1], color='g');
b = plt.scatter(data[0], data[1], color='b');
k = plt.scatter(data[0], data[1], color='k');
plt.show()