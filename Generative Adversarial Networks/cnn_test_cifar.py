import tensorflow as tf
from cnn import *
from read_cifar10 import *

def f(x):
	return x

network_architecture = { \
	'num_layers': 9, \
	'architecture': [\
						('conv', 64, 5, 3, tf.nn.relu), \
						('pool', 'MAX', 2), \
						('conv', 96, 4, 64, tf.nn.relu), \
						('pool', 'MAX', 2), \
						('conv', 128, 3, 96, tf.nn.relu), \
						('pool', 'MAX', 2), \
						('fc',  2048, 1024, tf.nn.relu), \
						('fc',  1024, 128, tf.nn.relu), \
						('fc', 128, 10, f) \
					], \
	'input_dim': (32, 32, 3), \
	'num_classes': 10, \
	'batch_size': 64, \
	'alpha': 1e-4, \
	'keep_prob': 0.7, \
	'num_epochs': 10 \
}

network = CNN(network_architecture)
cifar = CIFAR10('./cifar-10')

num_batches = int(cifar.get_num_training_examples() / network_architecture['batch_size'])
display_step = 1
save_step = 1

"""
num_batches = int(cifar.get_num_test_examples() / network_architecture['batch_size'])
acc = 0.0


for i in range(num_batches):
	inputs, labels = cifar.get_data(network_architecture['batch_size'], is_training = False)
	acc += network.test(inputs, labels)
print 'Accuracy: ' + str(acc / num_batches)

exit()
"""

for i in range(network_architecture['num_epochs']):
	avg_cost = 0.0
	for j in range(num_batches):
		inputs, labels = cifar.get_data(network_architecture['batch_size'])
		cost = network.train(inputs, labels)
		avg_cost += cost / cifar.get_num_training_examples() * network_architecture['batch_size']
		
	if i % display_step == 0:
		print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost))
		inputs, labels = cifar.get_data(network_architecture['batch_size'], is_training = False)
		print 'Accuracy: ' + str(network.test(inputs, labels))
		
	if i % save_step == 0:
		print 'Saving Checkpoint...'
		network.save_network()
