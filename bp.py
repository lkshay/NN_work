#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
#from src.activation import sigmoid, sigmoid_prime
from activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weights, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.
        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network
        Returns:
            (nabla_b, nabla_w): tuple containing the gradient for all the biases
                and weights. nabla_b and nabla_w should be the same shape as 
                input biases and weights
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]

    ### Implement here

    #Set up the weights and biases for the network
    w1 = np.zeros((len(weights[0][0]),len(weights[0])))
    print(w1.shape)
    b1 = np.zeros((len(biases[0]),1))
    print(b1.shape)
    w2 = np.zeros((len(weights[1][0]),len(weights[1])))
    print(w2.shape)
    b2 = np.zeros((len(biases[1]),1))
    print(b2.shape)

    # Implement feedforward pass
    # Here you need to store all the activations of all the units
    # by feedforward pass
    
    
    for i in range(w1.shape[1]):
        for j in range(w1.shape[0]):
            w1[j,i] = weights[0][i][j]
    
    for i in range(w2.shape[1]):
        for j in range(w2.shape[0]):
            w2[j,i] = weights[1][i][j]

    for i in range(b1.shape[0]):
        b1[i] = biases[0][i]

    for i in range(b2.shape[0]):
        b2[i] = biases[1][i]
    
    h1 = sigmoid(np.dot(np.transpose(w1),x) + b1)
    h2 = sigmoid(np.dot(np.transpose(w2),h1) + b2)

    #activations[-1] = h2

    #print(w1)
    #print(weights[0][0][783])

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer    

    #delta = (cost).delta(activations[-1], y)
    delta = (cost).delta(h2, y)
    #print(delta)

    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###

    return (nabla_b, nabla_w)