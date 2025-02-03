import numpy as np

import random

class Network:
    def __init__(self, sizes):                                                          # Contains the sizes vector which has the number of neurons in each layer
        self.num_layers = len(sizes)                                                    # Number of layers
        self.sizes = sizes          
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]                         # Randomly inititates biases
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]      # Randomly initiates weights


    def feedforward(self, a):                                                           # Takes the input a as network input
        for b,w in zip(self.biases, self.weights):                                      # Iterates through bias and weight matrices
            a = sigmoid(np.dot(w,a) + b)

        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data = None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):                                             # Iterates over the entire data set this many times
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]

            for mini_batch in mini_batches:                                 # Updates network for every SGD iteration
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                print ("Epoch {0}: {1}/{2}".format(j,self.evaluate(test_data), n_test))

            else:
                print ("Epoch {0} complete".format(j))


    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)

            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]

        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]


    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x                  # Represents the first activation layer
        activations = [x]               # Stores the list of all activations in layers
        zs = []                         # Stores all z values (values preceding sigmoid activation)

        for b,w in zip(self.biases, self.weights):
            z = w @ activation + b
            zs.append(z)
            activation = sigmoid(z)     # The new activation layer is defined
            activations.append(activation)      # The new activation layer is appended

        # Backward pass
        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta             # Defines new weights and bias derivatives for first layer
        nabla_w[-1] = delta @ activations[-2].transpose()

        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = (self.weights[-l+1].transpose() @ delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l-1].transpose()
        
        return (nabla_b,nabla_w)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))