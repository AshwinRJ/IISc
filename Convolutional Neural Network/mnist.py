import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from keras import backend as K
from keras.layers import Dense, Convolution2D, BatchNormalization, Dropout, Activation, Reshape

inputs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
labels = tf.placeholder(tf.float32, shape=(None, 10))

l = Convolution2D(nb_filter=16, nb_row=3, nb_col=3, init='he_normal', border_mode='same', subsample=(1, 1))(inputs)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = Dropout(0.2)(l)
l = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, init='he_normal', border_mode='same', subsample=(1, 1))(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = Dropout(0.2)(l)
l = Convolution2D(nb_filter=32, nb_row=1, nb_col=1, init='he_normal', border_mode='same', subsample=(2, 2))(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = Dropout(0.2)(l)
l = Reshape((6272,))(l)
l = Dense(400)(l)
l = Activation('relu')(l)
l = Dropout(0.2)(l)
l = Dense(10)(l)
output = Activation('sigmoid')(l)

network_cost = -tf.reduce_mean(labels*tf.log(output) + (1.0-labels)*tf.log(1.0-output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(network_cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(output), labels), tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
K.set_session(sess)

def train(x, y):
	_, batch_cost = sess.run((train_step, network_cost), feed_dict={inputs:x, labels:y, K.learning_phase(): 1})
	return batch_cost

def test(x, y):
	return sess.run(accuracy, feed_dict={inputs:x, labels:y, K.learning_phase(): 0})
	

mnist = input_data.read_data_sets('./mnist', one_hot=True)
num_batches = int(mnist.train.num_examples / 128)

display_step = 1

for i in range(1000):
	avg_cost = 0.0
	for j in range(num_batches):
		x, y = mnist.train.next_batch(128)
		x = np.reshape(x, [-1, 28, 28, 1])
		cost = train(x, y)
		avg_cost += cost / mnist.train.num_examples * 128
		
	if i % display_step == 0:
		print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost))
		x = mnist.test.images
		y = mnist.test.labels
		x = np.reshape(x, [-1, 28, 28, 1])
		print 'Accuracy:', test(x, y), 'Cost:', avg_cost
