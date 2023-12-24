# Neural Network Project

## Overview
This project implements a neural network for image classification using the MNIST dataset. The network is designed based on principles outlined in Michael Nielsen's book "Neural Networks and Deep Learning".

## Features
- **Neural Network Class**: Implements a basic feedforward neural network with customizable layers and neurons.
- **Stochastic Gradient Descent (SGD)**: Trains the network using mini-batch SGD.
- **Binary Evaluation**: In addition to standard evaluation, the network supports a binary evaluation mode, converting the network's final layer outputs into a binary representation for classification.
- **Dynamic Weight Adjustment**: The network dynamically adjusts its weights and biases based on the backpropagation algorithm during training.
- **Loss Visualization**: After training, the network can plot its performance over epochs.

## Usage
1. **Initialization**: Create an instance of the `Network` class, specifying the architecture.
2. **Training**: Train the network using the `StochasticGD` method with the desired parameters.
3. **Evaluation**: The network can be evaluated in two modes:
   - Standard mode, where the highest activation in the final layer determines the classification.
   - Binary evaluation mode, where the final layer's output is converted into a binary representation for classification.
4. **Plotting Loss**: After training, call `plot_loss` to visualize the training performance.

## Key Components
- `Network`: The main class representing the neural network.
- `StochasticGD`: Method for training the network with mini-batch stochastic gradient descent.
- `binary_evaluate`: A unique method that evaluates the network's performance using binary classification.
- `update_mini_batch`: Updates network weights and biases using gradient descent.
- `backprop`: Implements the backpropagation algorithm.
- `sigmoid` and `sigmoid_prime`: Activation function and its derivative.

## Dependencies
- Numpy
- Matplotlib
- MNIST Loader (included in the project)

## References
Based on 'Neural Networks and Deep Learning' by Michael Nielsen
- Nielsen, M. A. (2015). Neural Networks and Deep Learning. Determination Press.
- URL: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)
