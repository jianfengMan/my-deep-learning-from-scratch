import numpy as np

import matplotlib.pyplot as plt
import sys, os
from PIL import Image
import wx
import pickle

from setuptools._vendor.pyparsing import Dict

import MnistTest
from common.functions import sigmoid_grad, softmax, sigmoid

sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir, 'BookSource'))
from dataset.mnist import load_mnist


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def _cross_entropy_error(y, t):
    d = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + d)) / batch_size


def tangent(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d * x
    return lambda t: d * t + y


def _numerical_diff_num(f, x):
    h = 1e-7
    return (f(x + h) - f(x - h)) / (2 * h)


def _numerical_diff_array(f, x):
    h = 1e-7
    return np.array([(f(t + h) - f(t - h)) / (2 * h) for t in x])


def numerical_diff(f, x):
    if isinstance(x, float) or isinstance(x, int):
        return _numerical_diff_num(f, x)

    return _numerical_diff_array(f, x)


def numerical_gradient(f, x):
    grad = np.array([numerical_diff(f, t) for t in x])
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        print('x:', x)
        x -= lr * grad

    return x


def fun_2(x):
    r = np.sum(x ** 2)
    return r


def fun_1(x):
    return x ** 2


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        a = (np.sum(y == t) / float(x.shape[0]))
        return a

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = Dict.empty()
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


def trainNetwork():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    print('Begin Train!')
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = net.gradient(x_batch, t_batch)

        for key in ('W1', 'b1', 'W2', 'b2'):
            net.params[key] -= learning_rate * grad[key]

        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("Process:{:.0f}%, TrainAcc:{:.2f}%,, TestAcc:{:.2f}%".format((i / iters_num) * 100, train_acc * 100, test_acc * 100))

    return net


def getNetwork(fileName):
    try:
        netPickle = open(fileName, 'rb')
        net = pickle.load(netPickle)
    except FileNotFoundError:
        net = trainNetwork()
        netPickle = open(fileName, 'wb')
        pickle.dump(net, netPickle)

    netPickle.close()
    return net


if __name__ is '__main__':
    app = wx.App()
    frame = MnistTest.MnistTestWindow(getNetwork('mnist_network.nur'))
    frame.Show()
    app.MainLoop()