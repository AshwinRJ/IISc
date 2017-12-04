from GAN import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../Datasets/cifar-10/')
from read_cifar10 import *



initial_learning_rate1 = 0.0025
decay1 = 0.95
num_decay_steps1 = 150
batch1 = tf.Variable(0)
learning_rate1 = tf.train.exponential_decay( \
	initial_learning_rate1, \
	batch1, \
	num_decay_steps1, \
	decay1, \
	staircase=True \
)

initial_learning_rate2 = 0.0025
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
arch_gen['arch'] = [(100, None), (8000, tf.nn.relu, 0.05), (8000, tf.nn.relu, 0.05), (3072, tf.identity, 0.5)]
arch_gen['alpha'] = learning_rate1
arch_gen['keep_prob'] = 1

arch_dis = dict()
arch_dis['num_layers'] = 4
arch_dis['arch'] = [(3072, None), (1600, tf.nn.relu, 0.005), (1600, tf.nn.relu, 0.005), (1, tf.nn.sigmoid, 0.005)]
arch_dis['alpha'] = learning_rate2
arch_dis['keep_prob'] = 1

params = dict()
params['batch_size'] = 100
params['num_epochs'] = 200
	

class NoiseDistribution(object):
	# Distribution for generator input noise - uniform in given range

	def __init__(self, minval=-1, maxval=1):
		self.minval = minval
		self.maxval = maxval
	
	def sample(self, n=100):
		# Returns n samples from the distribution
		return np.random.uniform(self.minval, self.maxval, (n, 100))


cifar = CIFAR10('../Datasets/cifar-10/')

noise_dis = NoiseDistribution()

gan = GAN(arch_gen, arch_dis, params, noise_dis)

num_batches = int(cifar.get_num_training_examples() / params['batch_size'])
display_step = 1
save_step = 5

"""
for i in range(params['num_epochs']):
	d_avg_cost = 0.0
	g_avg_cost = 0.0
	for j in range(num_batches):
		inputs, labels = cifar.get_data(params['batch_size'])
		inputs = np.reshape(inputs, [-1, 3072])	# Flatten out images
		g_cost, d_cost = gan.train(inputs)
		d_avg_cost += d_cost / cifar.get_num_training_examples() * params['batch_size']
		g_avg_cost += g_cost / cifar.get_num_training_examples() * params['batch_size']
		
		if j % display_step == 0:
			print('\rBatch: ' + str(j+1)),
			sys.stdout.flush()
		
	if i % display_step == 0:
		print("Epoch:", '%04d' % (i+1), "Generator Cost=", "{:.4f}".format(g_avg_cost), "Discriminator Cost=", "{:.4f}".format(d_avg_cost))
		
	if i % save_step == 0:
		gan.save_network()
		
"""

sample = gan.generate(num_samples = 10)

for i in range(10):
	plt.imshow(np.reshape(sample[i, :], (32, 32, 3)))
	plt.show()
	
