# coding:utf-8
__author__ = 'fz'
__date__ = '2017-05-18 11:51'

from NeuralNetwork import NeuralNetwork
import numpy as np

#几层，每层分别有几个神经元[2,2,1] 2输入层
nn = NeuralNetwork([2, 2, 1], 'logistic')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, nn.predict(i))