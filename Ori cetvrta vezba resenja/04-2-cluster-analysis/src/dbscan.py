import numpy as np


class Cluster(object):

    def __init__(self):
        self.data = []  # podaci koji pripadaju ovom klasteru


class DBScan(object):

    def __init__(self, epsilon, min_points):
        """
        :param epsilon: za epsilon okolinu
        :param min_points: minimalan broj tacaka unutar epsilon okoline
        :return: None
        """
        self.epsilon = epsilon
        self.min_points = min_points
        self.data = None
        self.clusters = []

    def fit(self, data):
        self.data = data
        # TODO 6: implementirati DBSCAN
        # kada algoritam zavrsi, u self.clusters treba da budu klasteri (tipa Cluster)
        niz = np.array(data)
        self.data = niz.tolist()

        for tacka in self.data:
            tacka.append('not visited')

        cluster_index = -1
        for tacka in self.data:
            if 'visited' == tacka[-1]:
                continue

            tacka[-1] = 'visited'

            neighbors = self.get_neighbors(tacka)

            if len(neighbors) < self.min_points:
                tacka[-1] = 'noise'

            else:
                self.clusters.append(Cluster())
                cluster_index +=1
                self.expand_cluster(tacka,neighbors,cluster_index)

    def expand_cluster(self, point, neighbors, cluster_no):
        # Dodamo tacku 'point' u klaster sa rednim brojem 'claster_no'
        self.clusters[cluster_no].data.append(point[:-1])

        # Za svaku tacku u skupu susjeda
        for pt in neighbors:
            # Ako nije oznacena - oznacimo je
            if 'visited' != pt[-1]:
                pt[-1] = 'visited'

                # Dobavimo njene komsije
                neighbors_pts = self.get_neighbors(pt)

                # Ako ima minimalan broj komsija
                if len(neighbors_pts) >= self.min_points:
                    # Spojimo liste
                    neighbors.extend(neighbors_pts)

            # Provjeravamo da li se tacka nalazi u nekom klasteru
            point_in_cluster = False
            for c in self.clusters:
                for cluster_point in c.data:
                    if pt[:-1] == cluster_point:
                        point_in_cluster = True
                        break

                if point_in_cluster:
                    break

            # Ako tacka nije ni u jednom klasteru dodamo je
            if not point_in_cluster:
                self.clusters[cluster_no].data.append(pt[:-1])



    def get_neighbors(self,tacka):
        tacke = []
        tacke.append(tacka)

        for t in self.data:
            if self.euclidean_distance(t[:-1],tacka[:-1]) < self.epsilon:
                tacke.append(t)


        return tacke

    def euclidean_distance(self, x, y):
        sq_sum = 0
        for xi, yi in zip(x, y):
            sq_sum += (yi - xi) ** 2

        # Vracamo sqrt(sq_sum)
        return sq_sum ** 0.5

