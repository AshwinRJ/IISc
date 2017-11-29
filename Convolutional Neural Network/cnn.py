import tensorflow as tf
import os.path

class CNN:
	"""
		Implementation of CNN
	"""
	def __init__(self, network_architecture):
		"""
			__init__(CNN, dict) -> None
			network_architecture supports following keys:
				1. num_layers: Number of layers
				2. architecture: list of tuples (layer_type, layer_params) in order from input
								 to output layer.
								 layer_params are:
								 	2.1. num_filters, filter_dim, channel_in, activation function for
								 		 Convolutional Layers
								 	2.2. pool_type, filter_dim for Pooling Layers
								 	2.3. prev_num_units, num_units, activation_function for Fully Connected Layers
				3. input_dim: Dimension of inputs (input_width x input_height x num_channels)
				4. num_classes: Number of output classes
				5. batch_size: Batch size for the training of network
				6. alpha: Learning Rate for the network
				7. keep_prob: Dropout probability - Use 1 for no dropout
				8. num_epochs: Number of epochs through dataset
		"""
		self.num_layers = network_architecture['num_layers']
		self.arch = network_architecture['architecture']
		self.learning_rate = network_architecture['alpha']
		self.keep_prob = network_architecture['keep_prob']
		self.input_dim = network_architecture['input_dim']
		self.num_classes = network_architecture['num_classes']
		
		self.sess = tf.InteractiveSession()
		self.inputs, self.labels, self.weights = self._init_graph()
		layer_outputs = self.forward_pass()
		self.outputs = layer_outputs[self.num_layers - 1] # Collect the final outputs
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.outputs, 1), \
								tf.argmax(self.labels, 1)), tf.float32))
		self.cost = self.calculate_cost()	# Calculate the cost of the 	
		self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
			
		self.saver = tf.train.Saver()
		
		if (os.path.isfile('./Model/cnn-model.meta')):
			self.saver.restore(self.sess, './Model/cnn-model')
		
		else:
			self.sess.run(tf.global_variables_initializer())
		
		
	
	def _get_weight(self, shape):
		return tf.Variable(tf.truncated_normal(shape, stddev=0.01))
	
	def _get_bias(self, shape):
		return tf.Variable(tf.zeros(shape))
		
	
	def _init_graph(self):
		"""
			Initialize the inputs and outputs of the graph along with weights
		"""
		inputs = tf.placeholder(tf.float32, [None] + [x for x in self.input_dim])
		labels = tf.placeholder(tf.float32, [None, self.num_classes])
		
		weights = dict()
		k = 0
		
		for layer in self.arch:
			if layer[0] == 'conv':
				_, channel_out, filter_dim, channel_in, _ = layer
				weights['w' + str(k)] = self._get_weight((filter_dim, filter_dim, channel_in, channel_out))
				weights['b' + str(k)] = self._get_bias((channel_out))
			elif layer[0] == 'fc':
				_, prev_num_units, num_units, _ = layer
				weights['w' + str(k)] = self._get_weight((prev_num_units, num_units))
				weights['b' + str(k)] = self._get_bias((num_units))
			k += 1
			
		
		return (inputs, labels, weights)
		
	
	
	def forward_pass(self):
		"""
			Implement the feedforward pass on the network
		"""
		inputs = self.inputs
		outputs = dict()
		unroll = True	# Unrolling of data required for fully connected layers
		k = 0
		
		for layer in self.arch:
			if layer[0] == 'conv':
				outputs[k] = self._conv(inputs, k)
			elif layer[0] == 'pool':
				outputs[k] = self._pool(inputs, k)
			else:
				if (unroll):
					unroll = False
					inputs = tf.reshape(inputs, (-1, layer[1]))
				outputs[k] = self._fc(inputs, k)
			
			inputs = outputs[k]	# Prepare inputs for next layer
			k += 1
		
		return outputs
	
	
	
	def _conv(self, inputs, k):
		"""
			Perform the input-output mapping for kth (convolutional) layer
		"""
		w = self.weights['w' + str(k)]
		b = self.weights['b' + str(k)]
		activation_fn = self.arch[k][-1]
		conv = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], padding='SAME')
		activations = activation_fn(tf.nn.bias_add(conv, b))
		return activations
		
		

	def _pool(self, inputs, k):
		"""
			Perform the input-output mapping for kth (pooling) layer
		"""
		pool_dim = self.arch[k][2]
		if self.arch[k][1] == 'MEAN':	# Mean Pooling
			return tf.nn.avg_pool(inputs, ksize=[1, pool_dim, pool_dim, 1], \
							strides=[1, pool_dim, pool_dim, 1], padding='SAME')
		else:	# Max Pooling
			return tf.nn.max_pool(inputs, ksize=[1, pool_dim, pool_dim, 1], \
							strides=[1, pool_dim, pool_dim, 1], padding='SAME')
					
	
	
	def _fc(self, inputs, k):
		"""
			Perform the input-output mapping for kth (fully connected) layer
		"""
		w = self.weights['w' + str(k)]
		b = self.weights['b' + str(k)]
		activation_fn = self.arch[k][-1]
		if (k != self.num_layers - 1):
			return tf.nn.dropout(activation_fn(tf.matmul(inputs, w) + b), self.keep_prob)
		else:
			return tf.matmul(inputs, w) + b
	
	
	def calculate_cost(self):
		"""
			Calculate the cost using cross entropy cost function
		"""
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.outputs, self.labels))
		


	def train(self, x, y):
		"""
			Run network training with the given batch
		"""
		_, cost = self.sess.run((self.train_step, self.cost), feed_dict={self.inputs: x, self.labels: y})
		return cost
		
		
	
	def predict(self, x):
		"""
			Perform feedforward pass with given inputs
		"""
		outputs = self.sess.run(self.outputs, feed_dict={self.inputs: x})
		return outputs
	
	
	
	def test(self, x, y):
		temp = self.keep_prob	# Turn off dropout
		self.keep_prob = 1.0
		acc = self.sess.run(self.accuracy, feed_dict={self.inputs: x, self.labels: y})
		self.keep_prob = temp	# Turn dropout on again
		return acc
		
	
	def save_network(self):
		self.saver.save(self.sess, './Model/cnn-model')
