# coding:utf-8
__author__ = 'fz'
__date__ = '2017-04-24 16:28'

from numpy import *
import operator
import math


# def createDataSet():
#     # group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels
#
#
# def classify0(inX, dataSet, labels, k):
#     dataSetSize = dataSet.shape[0]
#     diffMat = tile(inX, (dataSetSize, 1)) - dataSet
#     sqDiffMat = diffMat**2
#     sqDistances = sqDiffMat.sum(axis=1)
#     distances = sqDistances**0.5
#     sortedDistIndicies = distances.argsort()
#     classCount={}
#     for i in range(k):
#         voteIlabel = labels[sortedDistIndicies[i]]
#         classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
#     sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
#     return sortedClassCount[0][0]
def ComputeEuclideanDistance(x1, y1, x2, y2):
    d = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    return d

d_ag = ComputeEuclideanDistance(3, 104, 18, 90)
d_bg = ComputeEuclideanDistance(2, 100, 18, 90)
d_cg = ComputeEuclideanDistance(1, 81, 18, 90)
d_dg = ComputeEuclideanDistance(101, 10, 18, 90)
d_eg = ComputeEuclideanDistance(99, 5, 18, 90)
d_fg = ComputeEuclideanDistance(98, 2, 18, 90)