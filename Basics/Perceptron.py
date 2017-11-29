#Prediction Function
def predict (input_row,weights):
    activation = weights[0]                        		# Bais is weights[0]
    for i in range(len(input_row)-1):
        activation += weights[i + 1] * input_row[i]
    return 1.0 if activation >= 0.0 else 0.0       		# Using a step function as transfer function

#Estimation of Perceptron Weights
def train_weights(train_data,l_rate,n_epochs):
    weights = [0.0 for i in range(len(train_data[0]))]      	# Initialized weights to zero
    
    for epoch in range(n_epoch):
        sum_error = 0.0                            		# To get total error in a particular epoch
        for row in train_data:
            prediction = predict (row,weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate*error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] + l_rate * error * row[i]
        print ">>Epoch=%d, Learning_rate = %0.4f, Error=%0.4f" %(epoch,l_rate,sum_error)
    return weights

# Training Preceptron using a training dataset
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1] ,
	[7.673756466,3.508563011,1]]
l_rate = 0.1
n_epoch = 5
weights = train_weights(dataset,l_rate,n_epoch)
print ">> Trained Weights:", weights               		# Final Updated Perceptron Weights

#Prediction for New data
new_data = [4.2344,5.3141]
prediction = predict(new_data,weights)
print "New data point belongs to class %d" %prediction