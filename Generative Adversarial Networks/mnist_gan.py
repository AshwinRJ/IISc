import tensorflow as tf
from GAN import *
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt



initial_learning_rate1 = 0.03
decay1 = 0.99
num_decay_steps1 = 150
batch1 = tf.Variable(0)
learning_rate1 = tf.train.exponential_decay( \
	initial_learning_rate1, \
	batch1, \
	num_decay_steps1, \
	decay1, \
	staircase=True \
)

initial_learning_rate2 = 0.001
decay2 = 0.9
num_decay_steps2 = 150
batch2 = tf.Variable(0)
learning_rate2 = tf.train.exponential_decay( \
	initial_learning_rate2, \
	batch2, \
	num_decay_steps2, \
	decay2, \
	staircase=True \
)

arch_gen = dict()
arch_gen['num_layers'] = 4
arch_gen['arch'] = [(100, None), (200, tf.nn.softplus), (400, tf.nn.softplus), (784, tf.nn.sigmoid)]
arch_gen['alpha'] = learning_rate1
arch_gen['keep_prob'] = 1

arch_dis = dict()
arch_dis['num_layers'] = 5
arch_dis['arch'] = [(784, None), (100, tf.nn.tanh), (60, tf.nn.tanh), (30, tf.nn.tanh), (1, tf.nn.sigmoid)]
arch_dis['alpha'] = learning_rate2
arch_dis['keep_prob'] = 1

params = dict()
params['batch_size'] = 128
params['num_epochs'] = 200
	

class NoiseDistribution(object):
	# Distribution for generator input noise - uniform in given range

	def __init__(self, minval=-1, maxval=1):
		self.minval = minval
		self.maxval = maxval
	
	def sample(self, n=100):
		# Returns n samples from the distribution
		return np.random.uniform(self.minval, self.maxval, (n, 100))


mnist = input_data.read_data_sets('./mnist', one_hot=True)

noise_dis = NoiseDistribution()

gan = GAN(arch_gen, arch_dis, params, noise_dis)

num_batches = int(mnist.train.num_examples / params['batch_size'])
display_step = 1

for i in range(params['num_epochs']):
	d_avg_cost = 0.0
	g_avg_cost = 0.0
	for j in range(num_batches):
		inputs, labels = mnist.train.next_batch(params['batch_size'])
		g_cost, d_cost = gan.train(inputs)
		d_avg_cost += d_cost / mnist.train.num_examples * params['batch_size']
		g_avg_cost += g_cost / mnist.train.num_examples * params['batch_size']
		
	if i % display_step == 0:
		print("Epoch:", '%04d' % (i+1), "Generator Cost=", "{:.4f}".format(g_avg_cost), "Discriminator Cost=", "{:.4f}".format(d_avg_cost))
		

sample = gan.generate(num_samples = 10)

for i in range(10):
	plt.imshow(np.reshape(sample[i, :], (28, 28)))
	plt.show()
	
