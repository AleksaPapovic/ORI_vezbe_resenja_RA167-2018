# TODO 3: implementirati primenu jednostavne linearne regresije
# nad podacima iz datoteke "data/skincancer.csv".
from __future__ import print_function
import pandas as pd

import csv
import linreg_simple as lrs
import matplotlib.pyplot as plt
def read_csv_file(filepath):

    # Ako je geotype = "Long" tada je u pitanju geografska duzina
    # Ako je geotype = "Lat"  tada je u pitanju geografska sirina
    x = []
    # Stopa smrtnosti - Mort
    y = []

    with open(filepath, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # Preskacemo header
        for row in reader:
             y.append(float(row[31]))
             x.append(int(row[28]))
    return x, y

def linreg_prvi(broj):
    x, y = read_csv_file('../data/dataset1.csv')
    slope, intercept = lrs.linear_regression(x, y)
    line_y = lrs.create_line(x, slope, intercept)
    for i in broj:
        rezultat = lrs.predict(i,slope,intercept)
        print("sa  " + str(i) + "  ubijenih ima popularnost  " + str(rezultat))
    plt.plot(x, y, '.')
    plt.plot(x, line_y, 'r')
    plt.title('Slope: {0}, intercept: {1}'.format(slope, intercept))
    plt.show()

if __name__ == '__main__':
    linreg_prvi(['5','10','15'])


