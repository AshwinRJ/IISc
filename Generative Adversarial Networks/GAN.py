# General Implementation of GANs

import tensorflow as tf
import numpy as np
import os

class GAN(object):
	
	def __init__(self, arch_gen, arch_dis, params, noise_dist):
		"""
			__init__(GAN, dict, dict, dict, noise_dist) -> None
			arch_gen and arch_ supports the following keys:
				1. num_layers: Number of layers
				2. arch: list of tuples (num_units, activation_fn)
				5. alpha: Learning rate for network
				6. keep_prob: Dropout probability - Use 1 for no dropout
		
			params supports the following keys:
				1. batch_size
				2. num_epochs
			
			noise_dist is a class which supports the following function:
			sample(num_samples) -> num_samples of size num_inputs_to_discriminator each
		"""
	
		self.arch_gen = arch_gen	# Architecture for generator network
		self.arch_dis = arch_dis	# Architecture for discriminator network
		self.params = params	# General parameters
		self.input_size = self.arch_dis['arch'][0][0]	# Number of input units in discriminator network
		self.noise_dist = noise_dist;	# Noise Generating Distribution for Generator
		self.batch_size = self.params['batch_size']
	
		# Initialize the weights and inputs
		self.inputs = tf.placeholder(tf.float32, shape = [None, self.input_size])	# Inputs to discriminator 
		self.weights_gen = self._init_weights(self.arch_gen, 'G')
		self.weights_dis = self._init_weights(self.arch_dis, 'D')
	
		# Run feedforward pass
		self.g_output = self._generate()
		self.d_output_real = self._discriminate(self.inputs)
		self.d_output_fake = self._discriminate(self.g_output)
	
		# Calculate the cost
		self.g_cost = tf.reduce_mean(-tf.log(self.d_output_fake))
		self.d_cost = tf.reduce_mean(-tf.log(self.d_output_real) - tf.log(1-self.d_output_fake))
	
		# Train the generator
		self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')	# Generator variables
		self.g_train = tf.train.AdamOptimizer(self.arch_gen['alpha']).minimize(self.g_cost,  var_list=self.g_variables)
	
		# Train the Discriminator
		self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')	# Generator variables
		self.d_train = tf.train.AdamOptimizer(self.arch_dis['alpha']).minimize(self.d_cost,  var_list=self.d_variables)
	
		# Set up the session
		self.sess = tf.InteractiveSession()
		
		# Set up the saver
		self.saver = tf.train.Saver()		
		if (os.path.isfile('./Model/gan-model.meta')):
			self.saver.restore(self.sess, './Model/gan-model')
		
		else:
			self.sess.run(tf.global_variables_initializer())
	
	
	
	
	def _get_weight(self, shape, scope=None, stddev=0.01):
		with tf.variable_scope(scope):
			return tf.Variable(tf.random_normal(shape, stddev=stddev))
	
	def _get_bias(self, shape, scope=None):
		with tf.variable_scope(scope):
			return tf.Variable(tf.zeros(shape))
	
	def _init_weights(self, arch, scope):
		"""
			_init_weights(GAN, dict, str) -> dict
			Initializes the weights corresponding to the provided architecture
		"""
		# Initialize weights
		weights = dict()
		
		# Set up useful variables
		arch = arch['arch']	# Use only the layer information
		prev_num_units = arch[0][0]
		
		# Remove input layer
		arch = arch[1:]
		
		i = 0
		for layer in arch:
			current_num_units = layer[0]
			stddev = layer[2]
			if stddev == None:
				weights['w' + str(i)] = self._get_weight((prev_num_units, current_num_units), scope)
			else:
				weights['w' + str(i)] = self._get_weight((prev_num_units, current_num_units), scope, stddev)
			weights['b' + str(i)] = self._get_bias((current_num_units), scope)
			prev_num_units = current_num_units
			i += 1
		
		return weights
		
	
	
	def _feedforward(self, inputs, weights, arch, scope, is_training=True):
		"""
			_feedforward(GAN, input tensor, dict, dict, str) -> output tensor
			Executes the feedforwad pass on the network with specified weights, architecture and scope
		"""
		keep_prob = arch['keep_prob'] # For dropout
		arch = arch['arch']	# Use only layer information
		arch = arch[1:]	# Remove the input layer
		num_layers = len(arch)	# Number of layers
		
		# Make sure that inputs are float32
		inputs = tf.cast(inputs, tf.float32)
				
		with tf.variable_scope(scope):
			# Initialize layer outputs
			outputs = inputs
			for i in range(num_layers):
				w = weights['w' + str(i)]
				b = weights['b' + str(i)]
				activation_fn = arch[i][1]
				if i != num_layers-1:	# No dropout in last layer
					if is_training:
						outputs = tf.nn.dropout(activation_fn(tf.matmul(outputs, w) + b), keep_prob)
					else:
						outputs = activation_fn(tf.matmul(outputs, w) + b)
				else:
					outputs = activation_fn(tf.matmul(outputs, w) + b)
			
			return outputs
	
	
	
	def _cost_gen(self, discriminator_outputs):
		"""
			_cost_gen(GAN, output tensor) -> cost tensor
			Calculate cost for the generator network.
		"""
		
		return tf.reduce_mean(-tf.log(self.discriminator_outputs))
	
	
	
	def _cost_dis(self, real_outputs, fake_outputs):
		"""
			_cost_dis(GAN, output tensor, output tensor) -> Cost tensor
			Calculate cost for discriminator network
		"""
		return tf.reduce_mean(-tf.log(self.real_outputs) - tf.log(1-self.fake_outputs))
		
	
	
	def _generate(self, inputs = None, is_training=True, num_samples = None):
		"""
			_generate(GAN, bool, int) -> output tensor
			Runs the feedforward pass on the generator network
		"""
		if inputs == None:	# If no inputs have been provided generate a sample from noise distribution
			if (num_samples == None):
				num_samples = self.batch_size
			inputs = self.noise_dist.sample(num_samples)
		
		return self._feedforward(inputs, self.weights_gen, self.arch_gen, 'G', is_training)
		
		
		
	def _discriminate(self, inputs, is_training=True):
		"""
			_discriminate(GAN, bool) -> output tensor
			Runs the feedforward pass on the discriminator network
		"""
		return self._feedforward(inputs, self.weights_dis, self.arch_dis, 'D', is_training)
	
	
	
	def train(self, inputs):
		"""
			train(GAN, input tensor) -> (g_cost, d_cost)
			Runs the training for a single batch
		"""
		# Train the discriminator
		_, d_cost = self.sess.run([self.d_train, self.d_cost], feed_dict = {self.inputs: inputs})
		
		# Train the generator
		_, g_cost = self.sess.run([self.g_train, self.g_cost])
		
		return (g_cost, d_cost)
		
		
	
	def generate(self, inputs = None, num_samples = None):
		"""
			generate(input
			Runs the generator on specified inputs
		"""
		if inputs == None:
			return self.sess.run(self._generate(is_training = False, num_samples = num_samples))
		else:
			return self.sess.run(self._generate(inputs, is_training = False, num_samples = num_samples))
		
	
	def discriminate(self, inputs):
		"""
			Runs the discriminator network on specified inputs
		"""
		return self.sess.run(self._discriminate(inputs, is_training = False))
	
	
	def save_network(self):
		self.saver.save(self.sess, './Model/gan-model')
	



if __name__ == '__main__':

	class DataDistribution(object):
		# Defines a gaussian data distribution with specified mean and variance
	
		def __init__(self, mean=5, sigma=0.5):
			self.mean = mean
			self.sigma = sigma
		
		def sample(self, n=100):
			# Returns n values sampled from the distribution
			return np.random.normal(self.mean, self.sigma, (n, 1))



	class NoiseDistribution(object):
		# Distribution for generator input noise - uniform in given range
	
		def __init__(self, minval=-1, maxval=1):
			self.minval = minval
			self.maxval = maxval
		
		def sample(self, n=100):
			# Returns n samples from the distribution
			return np.random.uniform(self.minval, self.maxval, (n, 1))
			

	initial_learning_rate = 0.03
	decay = 0.95
	num_decay_steps = 150
	batch = tf.Variable(0)
	learning_rate = tf.train.exponential_decay( \
		initial_learning_rate, \
		batch, \
		num_decay_steps, \
		decay, \
		staircase=True \
	)

	arch_gen = dict()
	arch_gen['num_layers'] = 3
	arch_gen['arch'] = [(1, None, None), (4, tf.nn.softplus, None), (1, tf.identity, None)]
	arch_gen['alpha'] = learning_rate
	arch_gen['keep_prob'] = 1
	
	arch_dis = dict()
	arch_dis['num_layers'] = 5
	arch_dis['arch'] = [(1, None, None), (8, tf.nn.tanh, None), (8, tf.nn.tanh, None), (8, tf.nn.tanh, None), (1, tf.nn.sigmoid, None)]
	arch_dis['alpha'] = learning_rate
	arch_dis['keep_prob'] = 1
	
	params = dict()
	params['batch_size'] = 12
	params['num_steps'] = 20000
		
	data_dist = DataDistribution()
	noise_dist = NoiseDistribution(minval=-8, maxval=8)
	gan = GAN(arch_gen, arch_dis, params, noise_dist)
	
	
	for i in range(params['num_steps']):
		inputs = data_dist.sample(params['batch_size'])
		g_cost, d_cost = gan.train(inputs)
		if i % 1000 == 0:
			print "Generator Cost: " + str(g_cost) + ", Discriminator Cost: " + str(d_cost)
	
	import matplotlib.pyplot as plt
	real_inputs = data_dist.sample(10000)
	fake_inputs = gan.generate(num_samples = 10000)
	plt.hist(real_inputs)
	plt.hist(fake_inputs)
	plt.show()
	 
