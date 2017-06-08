from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

'''
The MNIST data is split into three parts: 55,000 data points of training
data (mnist.train), 10,000 points of test data (mnist.test), and 5,000
points of validation data (mnist.validation). Each image is 28*28 = 784
'''

import tensorflow as tf

#define the shape of the input data.
x = tf.placeholder(tf.float32, [None,784])

#shape of model, one hidden layer with 10 neurons
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#foward
#sigmoid tf.sigmoid(x)
y = tf.nn.softmax(tf.matmul(x, W) + b)

#backward
#using cross-entropy to determine the loss. -sum(y',log(y))
y_ = tf. placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(
  y_*tf.log(y),reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
#initialize model
tf.global_variables_initializer().run()

#run 1000 epochs
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  #feed_dict is used to fill data into placeholder
  sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})

'''
tf.argmax gives you the index of the highest entry
in a tensor along some axis.
'''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy is: ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
