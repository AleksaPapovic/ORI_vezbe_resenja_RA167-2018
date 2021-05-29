from __future__ import print_function
import numpy, random, copy

class Cluster(object):

    def __init__(self, center):
        self.center = center
        self.data = []  # podaci koji pripadaju ovom klasteru

    def recalculate_center(self):
        # TODO 1: implementirati racunanje centra klastera
        # centar klastera se racuna kao prosecna vrednost svih podataka u klasteru


        novi_centar = [0 for i in range(len(self.center))]
        for datum in self.data:
            for i in range(len(datum)):
                novi_centar[i] += datum[i]
        n = len(self.data)

        if n!= 0:
            self.center = [x/n for x in novi_centar]


class KMeans2(object):

    def __init__(self, n_clusters, max_iter):
        """
        :param n_clusters: broj grupa (klastera)
        :param max_iter: maksimalan broj iteracija algoritma
        :return: None
        """
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []

    def fit(self, data, normalize = False):
        self.data = data  # lista N-dimenzionalnih podataka
        # TODO 4: normalizovati podatke pre primene k-means
        if normalize:
            self.data = self.normalize_data(self.data)
        # TODO 1: implementirati K-means algoritam za klasterizaciju podataka
        # kada algoritam zavrsi, u self.clusters treba da bude "n_clusters" klastera (tipa Cluster)
        dimenzija = len(self.data[0])

        def selection_sort(x):
            for i in range(len(x)):
                swap = i + numpy.argmin(x[i:])
                (x[i], x[swap]) = (x[swap], x[i])
            return x
        for i in range(self.n_clusters):
            #tacka = [random.random() for x in range(dimenzija)]
            indices = random.randrange(numpy.array(self.data).shape[0])
            tacka = self.data[numpy.array(indices)]
            self.clusters.append(Cluster(tacka))
        # TODO (domaci): prosiriti K-means da stane ako se u iteraciji centri klastera nisu pomerili
        iter_no = 0
        not_moves = False
        while iter_no <= self.max_iter and (not not_moves):
            # Ispraznimo podatke koji pripadaju klasteru
            for cluster in self.clusters:
                cluster.data = []

            for datum in self.data:
                # Nadjemo indeks klastera kom pripada tacka
                cluster_index = self.predict(datum)
                # Dodamo tu tacku u taj klaster da bismo mogli izracunati centar
                self.clusters[cluster_index].data.append(datum)



            # Preracunamo centar
            not_moves = True
            for cluster in self.clusters:
                old_center = copy.deepcopy(cluster.center)
                cluster.recalculate_center()

                #not_moves = not_moves and (cluster.center == old_center)

            #print("Iter no: " + str(iter_no))
            iter_no += 1

    def predict(self, datum):
        # TODO 1: implementirati odredjivanje kom klasteru odredjeni podatak pripada
        # podatak pripada onom klasteru cijem je centru najblizi (po euklidskoj udaljenosti)
        # kao rezultat vratiti indeks klastera kojem pripada
        min_distance = None
        cluster_index = None
        for index in range(len(self.clusters)):
            distance = self.euklidean_distance(datum, self.clusters[index].center)
            if min_distance is None or distance < min_distance:
                cluster_index = index
                min_distance = distance

        return cluster_index

    def euklidean_distance(self,x,y):
        sum = 0
        for xi, yi in zip(x,y):
            sum += (yi-xi)**2
        return  sum**0.5

    def normalize_data(self, data):
        # Broj kolona
        cols = len(data[0])

        for col in range(cols):

            # niz = numpy.asarray(data)
            # column_data = niz[0:2:len(niz)]

            column_data = []
            for row in data:
                column_data.append(row[col])

            # Izracunamo srednju vrijednost za kolonu
            mean = numpy.mean(column_data)
            # Izracunamo standardnu devijaciju za kolonu
            std = numpy.std(column_data)

            # Normalizujemo kolonu
            for row in data:
                row = numpy.asarray(row)
                row[col] =  (row[col] - mean) / std

            # new_list = []
            # for x, y in data:
            #     y = (y - mean) / std
            #     new_list.append((x, y))
            # data = new_list


        # Vratimo normalizovane podatke
        return data

    def sum_squared_error(self):
        # TODO 3: implementirati izracunavanje sume kvadratne greske
        # SSE (sum of squared error)
        # unutar svakog klastera sumirati kvadrate rastojanja izmedju podataka i centra klastera
        sse = 0
        for cluster in self.clusters:
            for datum in cluster.data:
                sse += self.euklidean_distance(cluster.center, datum)**2
        return  sse