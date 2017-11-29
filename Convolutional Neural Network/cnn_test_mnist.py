import tensorflow as tf
from cnn import *
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def f(x):
	return x

network_architecture = { \
	'num_layers': 6, \
	'architecture': [\
						('conv', 32, 5, 1, tf.nn.relu), \
						('pool', 'MAX', 2), \
						('conv', 64, 5, 32, tf.nn.relu), \
						('pool', 'MAX', 2), \
						('fc',  3136, 1024, tf.nn.relu), \
						('fc', 1024, 10, f) \
					], \
	'input_dim': (28, 28, 1), \
	'num_classes': 10, \
	'batch_size': 128, \
	'alpha': 1e-4, \
	'keep_prob': 0.5, \
	'num_epochs': 10 \
}

mnist = input_data.read_data_sets('./mnist', one_hot=True)

network = CNN(network_architecture)

num_batches = int(mnist.train.num_examples / network_architecture['batch_size'])
display_step = 1

for i in range(network_architecture['num_epochs']):
	avg_cost = 0.0
	for j in range(num_batches):
		inputs, labels = mnist.train.next_batch(network_architecture['batch_size'])
		inputs = np.reshape(inputs, [-1, 28, 28, 1])
		cost = network.train(inputs, labels)
		avg_cost += cost / mnist.train.num_examples * network_architecture['batch_size']
		
	if i % display_step == 0:
		print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost))
		inputs = mnist.test.images
		labels = mnist.test.labels
		inputs = np.reshape(inputs, [-1, 28, 28, 1])
		print 'Accuracy: ' + str(network.test(inputs, labels))
