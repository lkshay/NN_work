#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
'''
sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act
'''

import network2 as network2
import mnist_loader as loader
import activation as act


DATA_PATH = '../MNIST'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 50, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    epochs = 100
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    
    # model = network2.Network([784, 60, 10])
    model = network2.load("mymodel2.json")
    #train the network using SGD

    [evaluation_cost, evaluation_accuracy,training_cost, training_accuracy] = model.SGD(
        training_data=train_data,
        epochs=epochs,
        mini_batch_size=500,
        eta=1.9e-3,
        lmbda = 0.00001,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    epoch_list = []

    for i in range(epochs):
        epoch_list.append(i)

    # print((epoch_list),(training_cost))
    plt.figure()    
    plt.plot(epoch_list,training_cost,"r-")
    plt.ylabel("Training Cost")
    plt.xlabel("Epochs")
    plt.figure()
    plt.plot(epoch_list,training_accuracy,"r-")
    plt.ylabel("Training Accuracy")
    plt.xlabel("Epochs")
    plt.figure()
    plt.plot(epoch_list,evaluation_cost,"b-")
    plt.ylabel("Evaluation Cost")
    plt.xlabel("Epochs")
    plt.figure()
    plt.plot(epoch_list,evaluation_accuracy,"b-")
    plt.ylabel("Evaluation Accuracy")
    plt.xlabel("Epochs")
    
    model.save("mymodel2.json")
    
    # sleep(2)
    
    # trained_model = network2.load("mymodel.json")
    # test_result = trained_model.accuracy(test_data,convert = False)
    # print test_result

    plt.show()

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()