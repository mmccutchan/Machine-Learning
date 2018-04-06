import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def denseLayer(X, nodesOut, scope, activation=tf.nn.relu):
    with tf.variable_scope(scope if not None else 'FullyConnected'):
        Weights = tf.get_variable(name='Weights', shape=[X.get_shape()[1], nodesOut], dtype=tf.float32, initializer=tf.random_normal_initializer())
        Biases = tf.get_variable(name='Biases', shape=[nodesOut], dtype=tf.float32, initializer=tf.random_normal_initializer())
        return activation(X @ Weights + Biases)

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    X = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    h0 = denseLayer(X, 10, 'Layer1', tf.nn.softmax)

    loss = tf.losses.softmax_cross_entropy(y, h0)
    pred = tf.argmax(h0, 1)
    actual = tf.argmax(y, 1)
    correct = tf.equal(pred, actual)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), 0)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    epochs = 100
    batchSize = 100

    for epoch in range(epochs):
        for numBatches in range(mnist.train.images.shape[0] // batchSize):
            batch, labels = mnist.train.next_batch(batchSize)
            sess.run(optimizer, feed_dict={X:batch, y:labels})
        if epoch % 20 == 0:
            acc = sess.run(accuracy, feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
            currLoss = sess.run(loss, feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
            print("Accuracy at epoch " + str(epoch) + ": " + str(acc))
            print("Loss at epoch " + str(epoch) + ": " + str(currLoss))

    acc = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
    print("Final test accuracy: " + str(acc))
