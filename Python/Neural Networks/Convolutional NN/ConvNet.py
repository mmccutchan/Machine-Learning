import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def convLayer(x, filterSize, nodesIn, nodesOut, activation=None, scope=None):
    with tf.variable_scope('Convolution/' + scope):
        w = tf.get_variable('weights', dtype=tf.float32,
                            shape=[filterSize,filterSize, nodesIn, nodesOut],
                            initializer=tf.random_normal_initializer(stddev=0.1, mean=0.0))
        b = tf.get_variable('biases', dtype=tf.float32,
                            shape=[nodesOut],
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1,2,2,1], padding='SAME'), b)
        if activation is not None:
            h = activation(h)
        return h

def denseLayer(x, nodesIn, nodesOut, activation=None, scope=None):
    with tf.variable_scope('FullyConnected/' + scope):
        w = tf.get_variable('weights', dtype=tf.float32,
                            shape=[nodesIn, nodesOut],
                            initializer=tf.random_normal_initializer(stddev=0.1, mean=0.0))
        b = tf.get_variable('biases', dtype=tf.float32,
                            shape=[nodesOut],
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, w), b)
        if activation is not None:
            h = activation(h)
        return h

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    X = tf.placeholder(tf.float32, shape=[None, 784])
    XFlat = tf.reshape(X, [-1, 28, 28, 1])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    h0 = convLayer(XFlat, 5, 1, 32, tf.nn.relu, 'Layer1')
    h1 = convLayer(h0, 5, 32, 64, tf.nn.relu, 'Layer2')
    h2 = denseLayer(tf.reshape(h1, (-1, 49 * 64)), 49 * 64, 128, tf.nn.relu, 'Layer3')
    h3 = denseLayer(h2, 128, 10, tf.nn.softmax, 'Layer4')

    loss = tf.losses.softmax_cross_entropy(y, h3)
    pred = tf.argmax(h3, 1)
    actual = tf.argmax(y, 1)

    correct = tf.equal(pred, actual)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), 0)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    epochs = 10
    batchSize = 100

    for epoch in range(epochs):
        for numBatches in range(mnist.train.images.shape[0] // batchSize):
            batch, labels = mnist.train.next_batch(batchSize)
            sess.run(optimizer, feed_dict={X:batch, y:labels})
        acc = sess.run(accuracy, feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
        currLoss = sess.run(loss, feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
        print("Accuracy at epoch " + str(epoch) + ": " + str(acc))
        print("Loss at epoch " + str(epoch) + ": " + str(currLoss))

    acc = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
    print("Final test accuracy: " + str(acc))
