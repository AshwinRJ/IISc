import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Some useful variables
mean = 5;
sigma = 5;

class DataDistribution(object):
	# Defines a gaussian data distribution with specified mean and variance
	
	def __init__(self, mean=5, sigma=0.5):
		self.mean = mean
		self.sigma = sigma
		
	def sample(self, n=100):
		# Returns n values sampled from the distribution
		return np.random.normal(self.mean, self.sigma, (n, 1))



class GeneratorDistribution(object):
	# Distribution for generator input noise - uniform in given range
	
	def __init__(self, minval=-1, maxval=1):
		self.minval = minval
		self.maxval = maxval
		
	def sample(self, n=100):
		# Returns n samples from the distribution
		return np.random.uniform(self.minval, self.maxval, (n, 1))
		


class GAN(object):
	# Toy GAN implementation
	def __init__(self, hidden_size_gen, hidden_size_dis1, hidden_size_dis2, hidden_size_dis3):
		# Useful variables
		self.hidden_size_dis1 = hidden_size_dis1
		self.hidden_size_dis2 = hidden_size_dis2
		self.hidden_size_dis3 = hidden_size_dis3
		self.hidden_size_gen = hidden_size_gen
		self.data_dis = DataDistribution(4, 0.5)
		self.noise_dis = GeneratorDistribution(-8, 8)
		
		# Initialize weights
		self.weights_gen = dict()
		self.weights_dis = dict()
		self._init_weights()
		
		# Prepare inputs
		self.inputs_gen = tf.placeholder(tf.float32, shape = [None, 1])
		self.inputs_dis = tf.placeholder(tf.float32, shape = [None, 1])
		
		# Feedforward pass
		with tf.variable_scope('G'):
			self.g_outputs = self.generator(self.inputs_gen)
		with tf.variable_scope('D') as scope:
			self.d_outputs_real = self.discriminator(self.inputs_dis)
			scope.reuse_variables()
			self.d_outputs_fake = self.discriminator(self.g_outputs)
		
		# Calculate cost
		self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
		self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
		self.g_cost = tf.reduce_mean(-tf.log(self.d_outputs_fake))
		self.d_cost = tf.reduce_mean(-tf.log(self.d_outputs_real) - tf.log(1-self.d_outputs_fake))
		
		# Train
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
		self.g_train = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.g_cost, global_step=batch, var_list=self.g_variables)
		self.d_train = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.d_cost, global_step=batch, var_list=self.d_variables)
		
		# Set up session
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())
				
	
	
	def _get_weight(self, shape, scope=None):
		with tf.variable_scope(scope):
			return tf.Variable(tf.random_normal(shape, stddev=1.0))
	
	def _get_bias(self, shape, scope=None):
		with tf.variable_scope(scope):
			return tf.Variable(tf.zeros(shape))
	
	def _init_weights(self):
		# Initialize the weights
		self.weights_gen['w1'] = self._get_weight((1, self.hidden_size_gen), 'G')
		self.weights_gen['b1'] = self._get_bias((self.hidden_size_gen), 'G')
		self.weights_gen['w2'] = self._get_weight((self.hidden_size_gen, 1), 'G')
		self.weights_gen['b2'] = self._get_bias((1), 'G')
		# Discriminator weights
		self.weights_dis['w1'] = self._get_weight((1, self.hidden_size_dis1), 'D')
		self.weights_dis['b1'] = self._get_bias((self.hidden_size_dis1), 'D')
		self.weights_dis['w2'] = self._get_weight((self.hidden_size_dis1, self.hidden_size_dis2), 'D')
		self.weights_dis['b2'] = self._get_bias((self.hidden_size_dis2), 'D')
		self.weights_dis['w3'] = self._get_weight((self.hidden_size_dis2, self.hidden_size_dis3), 'D')
		self.weights_dis['b3'] = self._get_bias((self.hidden_size_dis3), 'D')
		self.weights_dis['w4'] = self._get_weight((self.hidden_size_dis3, 1), 'D')
		self.weights_dis['b4'] = self._get_bias((1), 'D')
	
	
	def generator(self, inputs):
		# Generate the outputs using the inputs
		w1 = self.weights_gen['w1']
		b1 = self.weights_gen['b1']
		h1 = tf.nn.softplus(tf.matmul(inputs, w1) + b1)
		
		w2 = self.weights_gen['w2']
		b2 = self.weights_gen['b2']
		outputs = tf.matmul(h1, w2) + b2
		
		return outputs
		
	
	def discriminator(self, inputs):
		# Decide if the input is real or fake
		w1 = self.weights_dis['w1']
		b1 = self.weights_dis['b1']
		h1 = tf.nn.tanh(tf.matmul(inputs, w1) + b1)
		
		w2 = self.weights_dis['w2']
		b2 = self.weights_dis['b2']
		h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)
		
		w3 = self.weights_dis['w3']
		b3 = self.weights_dis['b3']
		h3 = tf.nn.tanh(tf.matmul(h2, w3) + b3)
		
		w4 = self.weights_dis['w4']
		b4 = self.weights_dis['b4']
		outputs = tf.nn.sigmoid(tf.matmul(h3, w4) + b4)
		
		return outputs
		
	
	def train(self, num_steps=10000, batch_size=12):
		# Train using the provided data
		
		for step in range(num_steps):
			# Train discriminator
			noise = self.noise_dis.sample(batch_size)
			real_data = self.data_dis.sample(batch_size)		
			cost1, _ = self.sess.run([self.d_cost, self.d_train], \
										feed_dict={self.inputs_gen: noise, self.inputs_dis: real_data})
			
			# Train Generator
			noise = self.noise_dis.sample(batch_size)
			cost2, _ = self.sess.run([self.g_cost, self.g_train], feed_dict={self.inputs_gen: noise})
			
			if step % 1000 == 0:
				print "Generator Cost: " + str(cost1) + "\t" + str(cost2)
				
	
	def generate_data(self, num_samples):
		# Generates data from learned distribution
		noise = self.noise_dis.sample(num_samples)
		samples = self.sess.run(self.g_outputs, feed_dict={self.inputs_gen: noise})
		return samples


if __name__ == '__main__':
	gan = GAN(4, 8, 8, 8)
	gan.train(200000)
	samples = gan.generate_data(10000)
	real = gan.data_dis.sample(10000)
	
	plt.hist(samples)
	plt.hist(real)
	plt.show()
	
	print gan.sess.run(gan.weights_dis['w1'])
	print gan.sess.run(gan.weights_dis['b1'])
	print gan.sess.run(gan.weights_dis['w2'])
	print gan.sess.run(gan.weights_dis['b2'])
	
	print '***********'
	print gan.sess.run(gan.weights_gen['w1'])
	print gan.sess.run(gan.weights_gen['b1'])
	print gan.sess.run(gan.weights_gen['w2'])
	print gan.sess.run(gan.weights_gen['b2'])
	
	x = np.linspace(-10, 10, 100, dtype=np.float32)
	x = np.reshape(x, (-1, 1))
	y = gan.sess.run(gan.d_outputs_real, feed_dict={gan.inputs_dis: x})
	plt.plot(x, y)
	y = gan.sess.run(gan.g_outputs, feed_dict={gan.inputs_gen: x})
	plt.plot(x, y)
	plt.show()
