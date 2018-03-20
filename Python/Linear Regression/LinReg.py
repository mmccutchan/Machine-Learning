import numpy as np
import matplotlib.pyplot as plt
import csv

def computeCost(X, y, theta):
    return sum((X @ theta - y) ** 2)

def normalEquation(X, y):
    return np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

def gradientDescent(X, y, theta, learningRate = 0.01, epochs = 50):
    for epoch in range(epochs):
        theta = theta - np.reshape(learningRate / y.shape[0] * np.transpose(sum((X @ theta - y) * X)), (-1, theta.shape[1])) #y needs to be a numpy array
    return theta

def normalize(X):
    return (X - np.mean(X, 0)) / np.std(X, 0)

if __name__ == "__main__":
    X = []
    y = []
    with open(r"C:\Users\mmccutchan\source\repos\Machine-Learning-Algorithms\Python\Linear Regression\data2.txt") as file:
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
    X = normalize(X)

    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


    theta = normalEquation(X, y)
    print("Theta derived from normal equation\n" + str(theta))


    theta = np.zeros((X.shape[1],1))
    theta = gradientDescent(X, y, theta, 0.01, 1500)
    print("Theta derived from gradient descent\n" + str(theta))
