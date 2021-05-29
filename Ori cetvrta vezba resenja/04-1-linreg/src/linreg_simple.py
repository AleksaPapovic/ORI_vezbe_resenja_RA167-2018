import random
import matplotlib.pyplot as plt


def linear_regression(x, y):
    slope = 0.0  # nagib linije
    intercept = 0.0  # tacka preseka na y-osi
    # TODO 1: implementirati linearnu regresiju
    #n broj elemenata
    n = len(x)


    x_srednje = float(sum(x))/n
    y_srednje = float(sum(y))/n

    X_COV = [tx - x_srednje for tx in x]
    Y_COV = [ty - y_srednje for ty in y]

    # Σ (x-x_srednje)(y-y_srednje) / Σ (x-xsrednje)^2
    a = sum([i*j for i,j in zip(X_COV,Y_COV)]) / sum([i**2 for i in X_COV])

    b =  y_srednje  - a * x_srednje

    slope =  a
    intercept = b

    return slope, intercept


def predict(x, slope, intercept):
    # TODO 2: implementirati racunanje y na osnovu x i parametara linije
    y = slope*float(x) + intercept
    return y


def create_line(x, slope, intercept):
    y = [predict(xx, slope, intercept) for xx in x]
    return y


if __name__ == '__main__':
    x = range(50)  # celobrojni interval [0,50]
    random.seed(1337)  # da rezultati mogu da se reprodukuju
    y = [(i + random.randint(-5, 5)) for i in x]  # y = x (+- nasumicni sum)
    slope, intercept = linear_regression(x, y)
    line_y = create_line(x, slope, intercept)
    plt.plot(x, y, '.')
    plt.plot(x, line_y, 'b')
    plt.title('Slope: {0}, intercept: {1}'.format(slope, intercept))
    plt.show()
