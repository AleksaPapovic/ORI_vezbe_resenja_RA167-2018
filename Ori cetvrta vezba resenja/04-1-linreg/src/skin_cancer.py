# TODO 3: implementirati primenu jednostavne linearne regresije
# nad podacima iz datoteke "data/skincancer.csv".
from __future__ import print_function


import csv
import linreg_simple as lrs
import matplotlib.pyplot as plt
def read_csv_file(filepath, geotype = 'Long'):

    # Ako je geotype = "Long" tada je u pitanju geografska duzina
    # Ako je geotype = "Lat"  tada je u pitanju geografska sirina
    x = []
    # Stopa smrtnosti - Mort
    y = []

    index = 4 if geotype == "Long" else 1

    with open(filepath, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # Preskacemo header
        for row in reader:
            x.append(float(row[index]))
            y.append(int(row[2]))

    return x, y

def linreg_cancer(geotype):
    x, y = read_csv_file('../data/skincancer.csv', geotype)
    slope, intercept = lrs.linear_regression(x, y)
    line_y = lrs.create_line(x, slope, intercept)
    plt.plot(x, y, '.')
    plt.plot(x, line_y, 'r')
    plt.title('Slope: {0}, intercept: {1}'.format(slope, intercept))
    plt.show()


if __name__ == '__main__':
    linreg_cancer('Lat')

    linreg_cancer('Long')
