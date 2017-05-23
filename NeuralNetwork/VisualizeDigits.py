# coding:utf-8
__author__ = 'fz'
__date__ = '2017-05-18 14:24'

from sklearn.datasets import load_digits


digits = load_digits()
print digits.data.shape

import pylab as pl
pl.gray()
pl.matshow(digits.images[0])
pl.show()