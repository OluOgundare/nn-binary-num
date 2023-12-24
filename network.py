import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from mnist_loader import load_data_wrapper
import random

class Network():
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)

        # Sizes contains the number of neurons in the respective layers
        # i.e. for a 3 layer neuron of 1,2, and 3 neurons respectively, list will be [1,2,3]
        self.sizes = sizes

        # np.random.randn is a sample from a random normal gaussian distirbution
        # set of biases for each layer. y,1 denotes array sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # y,x denotes matrix with sizes y,x
        # Weights is a matrix such that Wjk denotes the connection between the kth neuron in
        # the 2nd layer and jth neuron in the third layer
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

        self.history = []

    def feedforward(self, a):
        """Returns the output of the network if "a" is an input"""
        for b, w in zip(self.biases, self.weights):
            # a' = sigmoid(w*a + b)
            a = self.sigmoid(np.dot(w, a) + b)
        self.final_layer = a
        return a
    
    def StochasticGD(self, training_data, epochs, mini_batch_size, eta, test_data = None, binary_eval = False):

        """Train the neural network using mini-batch stochastic gradient.
        The training data is a list of tuples (x,y) representing the training inputs and the 
        desired outputs. eta is the learning rate. If test_data is provided then the 
        network will be evaluated against the test data after each epoch and partial progress printed out."""

        # Training data is a list of tuples (x,y) representing the training data and the desired output

        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):

            # Randomly shuffle the train data 
            random.shuffle(training_data)

            mini_batches = [
                training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]

            # Apply a single step of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                eval = self.evaluate(test_data) if not binary_eval else self.binary_evaluate(test_data)
                print("Epoch {0} : {1} / {2}".format(j, eval, n_test))
            else:
                print(f"Epoch {j} complete")

            self.history.append(eval / n_test)

    def update_mini_batch(self, mini_batch, eta):
        """Update the networks weights and biases by applying gradient descent 
        using backpropagation to a single mini batch.
        Mini batch is a list of tuples (x,y) and eta is the learning rate"""


        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x ,y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # Apply backpropagation algorithm

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # Update biases based on gradient

            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # Update weights based on gradient

        self.weights = [w - (eta / len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta / len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return the tuples (nabla_b, nabla_w) representing the gradient for the cost function C_x. Nabla_b and
        nabla_w are layer-by-layer lists of numpy arrays similar to self.biases and self.weights"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x] # List to store all the activations layer by layer
        zs = [] # List to store all the z vectors layer by layer

        for b, w in zip(self.biases, self.weights):

            z = np.dot(w,activation) + b

            zs.append(z)

            activation = self.sigmoid(z)

            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) *  self.sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        """Note that the variable l = 1 means that the last layer of the neurons,
         l = 2 means the second to last and so on."""

        for l in range(2, self.num_layers):
            z = zs[-l]

            sp = self.sigmoid_prime(z)

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the numnber of test inputs for which the neural network outputs the currect result.
        Note that the neural network's output is assumed to be the index of whichever neuron in the final layer
        has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for x,y in test_data]
        return sum((x == y) for x, y in test_results)
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial Cx / partial a for the output activations"""
        return output_activations - y


    def sigmoid(self, z):
        # If z is  a vector, numpy applies the function element wise
        return 1.0 / (1.0 + np.exp(-z))
    

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def binary_evaluate(self, test_data):
        final_layers = deque([])
        outputs = []
        for x,y in test_data:
            outputs.append((self.feedforward(x), y))
            final_layers.append(self.final_layer)
        output_sum = 0
        binary_weights = np.matrix([
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

        for _, output in outputs:
            output = bin(output)
            binary_output = np.matmul(binary_weights, final_layers.popleft())
            binary_num = []
            for x in binary_output:
                if x > 0.7:
                    binary_num.append("1")
                elif binary_num:
                    binary_num.append("0")

            binary_num = "".join(binary_num)
            if binary_num == output[2:]: output_sum += 1
        return output_sum

    
    def plot_loss(self):
        plt.plot(self.history)
        plt.show()

train_d, val_d, test_d = load_data_wrapper()
neural_network = Network([784, 30, 10])

neural_network.StochasticGD(training_data = train_d, epochs = 20, mini_batch_size = 20, eta = 1.5, test_data = test_d,
                            binary_eval = True)

neural_network.StochasticGD(training_data = train_d, epochs = 20, mini_batch_size = 20, eta = 1.5, test_data = test_d,
                            binary_eval = False)