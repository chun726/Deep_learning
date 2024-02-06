"""
HW2 problem
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt


########################################
# cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return cross entropy loss
    softmax_score = None
    softmax_loss = 0
    for data in x:  # Iterate within x_data
        single_score = np.zeros((num_class))
        single_exp_score = np.zeros((num_class))
        exponential_sum = 0

        for i in range(num_class):  # Calculate linear score for each classes
            linear_sum = (
                data[0] * Wb[2 * i] + data[1] * Wb[2 * i + 1] + Wb[num_class * 2 + i]
            )

            single_score[i] = linear_sum  # [1, 3] vector of linear score (single point)
            exponential_sum += np.exp(linear_sum)  # Pre-generating exponential sum

        for i in range(num_class):
            single_exp_score[i] = np.exp(single_score[i]) / exponential_sum

        if softmax_score is None:
            softmax_score = single_exp_score
        else:
            softmax_score = np.vstack((softmax_score, single_exp_score))

    for i, elem in enumerate(softmax_score):
        softmax_loss -= np.log10(elem[y[i]])

    return softmax_loss


# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x, y, num_class):
    n_test = x.shape[0]
    feat_dim = x.shape[1]

    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:].squeeze()
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    accuracy = 0

    # W has shape (num_class, feat_dim), b has shape (num_class,)

    # score
    s = x @ W.T + b
    # score has shape (n_test, num_class)

    # get argmax over class dim
    res = np.argmax(s, axis=1)

    # get accuracy
    accuracy = (res == y).astype("uint8").sum() / n_test

    return accuracy


# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print("number of classes: ", num_class, " sigma for data scatter:", sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print("generating training data")
x_train, y_train = dg.generate(
    number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma
)

# generate test dataset
print("generating test data")
x_test, y_test = dg.generate(
    number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma
)

# start training softmax classifier
print("training softmax classifier...")
w0 = np.random.normal(0, 1, (2 * num_class + num_class))
result = minimize(
    cross_entropy_softmax_loss,
    w0,
    args=(x_train, y_train, num_class, n_train, feat_dim),
)

print("testing softmax classifier...")

Wb = result.x
print(
    "accuracy of softmax loss: ",
    linear_classifier_test(Wb, x_test, y_test, num_class) * 100,
    "%",
)

print(x_train)
