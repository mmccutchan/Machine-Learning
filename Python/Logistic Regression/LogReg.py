import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import os

def loadData(fileName):
    X = []
    y = []
    with open(r"C:\Users\mmccutchan\source\repos\Machine-Learning-Algorithms\Python\Logistic Regression" + "\\" + fileName) as file:
        reader = csv.reader(file)
        for row in reader:
            array = []
            for num in row[:-1]:
                array.append(float(num))
            X.append(array)

            array = []
            for num in row[-1:]:
                array.append(float(num))
            y.append(array)

    X = np.array(X)
    y = np.array(y)
    return (X, y)

def sigmoid(X):
    return 1 / (1 + math.e ** (-X))

def computeCost(X, y, theta, regCoeff):
    m = len(y)
    cost = 1 / m * np.sum(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta))) + regCoeff / 2 * np.sum(theta[1:] ** 2)
    return cost

def gradientDescent(X, y, theta, regCoeff=0, learningRate=0.01, epochs=50):
    costs = []
    for epoch in range(epochs):
        theta[1:] = theta[1:] -  learningRate / len(y) * np.reshape(np.sum((sigmoid(X @ theta) - y) * X, axis = 0) , theta.shape)[1:] - regCoeff / len(y) * theta[1:]
        theta[0] = theta[0] - learningRate / len(y) * np.sum((sigmoid(X @ theta) - y) * X, axis = 0)[0]
        costs.append(computeCost(X, y, theta, regCoeff))
    return (theta, costs)

def normalize(X):
    return (X - np.mean(X, 0)) / np.std(X, 0)

if __name__ == "__main__":
    X, y = loadData('data1.txt')
    X = normalize(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    theta = np.zeros((X.shape[1], 1))
    theta, costs = gradientDescent(X, y, theta, epochs=5)
    posX = []
    negX = []

    for i in np.argwhere(y == 0):
        negX.append(X[i[0]][1:])

    for i in np.argwhere(y == 1):
        posX.append(X[i[0]][1:])

    print("Theta derived from Logistic Gradient Descent:\n" + str(theta)) #Normalization dramatically improves this


    xBound = [np.min(X, axis=0)[1] - 2, np.max(X, axis=0)[1] + 2] #Plot decision boundary
    yBound = -1 / theta[2] * (theta[1] * xBound + theta[0])
    plt.plot(xBound, yBound)

    for point in posX: #Plot positive and negative examples
        plt.plot(point[0], point[1], 'bo')
    for point in negX:
        plt.plot(point[0], point[1], 'rx')
    plt.show()
