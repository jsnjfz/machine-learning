# coding:utf-8
__author__ = 'fz'
__date__ = '2017-05-11 16:48'

from sklearn import svm


X = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

print(clf)
