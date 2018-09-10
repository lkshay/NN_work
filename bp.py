#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
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

	#print(x.shape)
	#print(weights[0].shape)
	#print(biases[0].shape)

	### Implement here
	# feedforward
	# Here you need to store all the activations of all the units
	# by feedforward pass
	###

	h1 = sigmoid(np.dot(weights[0],x) + biases[0])
	h2 = sigmoid(np.dot(weights[1],h1) + biases[1])

	# compute the gradient of error respect to output
	# activations[-1] is the list of activations of the output layer
	delta = (cost).delta(h2, y)

	### Implement here
	# backward pass
	# Here you need to implement the backward pass to compute the
	# gradient for each weight and bias
	###

	nabla_b[1] = delta*sigmoid_prime(h2)
	nabla_w[1] = np.dot(nabla_b[1],h1.transpose())

	nabla_b[0] = np.dot(weights[1].transpose(),nabla_b[1])*sigmoid_prime(h1)	
	nabla_w[0] = np.dot(nabla_b[0],x.transpose())
	#nabla_b[0] = sigmoid_prime(h1)


	#print(weights[1].shape)
	#print(nabla_w[0].shape)
	#print(nabla_w[1].shape)	

	#print(np.dot(nabla_b[1],h1.transpose()).shape)
	#print(np.dot(nabla_b[0],x.transpose()).shape)
	 
	


	return (nabla_b, nabla_w)








