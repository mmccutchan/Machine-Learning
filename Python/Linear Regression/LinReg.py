import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D

def loadData(fileName):
    X = []
    y = []
    with open(r"C:\Users\mmccutchan\source\repos\Machine-Learning-Algorithms\Python\Linear Regression" + "\\" + fileName) as file:
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

def computeCost(X, y, theta):
    return sum((X @ theta - y) ** 2)

def normalEquation(X, y):
    return np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

def gradientDescent(X, y, theta, learningRate = 0.01, epochs = 50):
    costs = []
    for epoch in range(epochs):
        theta = theta - np.reshape(learningRate / y.shape[0] * np.transpose(sum((X @ theta - y) * X)), theta.shape) #y needs to be a numpy array
        costs.append(computeCost(X, y, theta))
    return (theta, costs)

def normalize(X):
    return (X - np.mean(X, 0)) / np.std(X, 0)

if __name__ == "__main__":
    X, y = loadData('data2.txt')
    print(X.shape, y.shape)
    X = normalize(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    theta = normalEquation(X, y)
    print("Theta derived from normal equation\n" + str(theta))


    theta = np.zeros((X.shape[1],1))
    theta, costs = gradientDescent(X, y, theta, 0.01, 1500)
    print("Theta derived from gradient descent\n" + str(theta))
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost over Iterations')
    plt.show()

    testX = np.linspace(0, 10000, 100)
    testY = np.linspace(0, 10, 100)
    xx, yy = np.meshgrid(testX, testY)
    z = xx * theta[1] + yy * theta[2] + theta[0]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Axes3D.plot_surface(ax, xx, yy, z)
    plt.xlabel('Square Footage')
    plt.ylabel('Number of Bedrooms')
    ax.set_zlabel('Cost of House')
    plt.show()
