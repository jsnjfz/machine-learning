# coding:utf-8
__author__ = 'fz'
__date__ = '2017-05-04 17:01'

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn import tree
from sklearn.externals.six import StringIO

import csv


allElectronicsData = open(r'E:\code\python\machine-learning\AllElectronics.csv')
reader = csv.reader(allElectronicsData)
headers = reader.next()

print(headers)

# 特征值列表
featureList = []

# class类别的列表
labelList = []

for row in reader:
    #加入最后一列
    # print (row)
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
        # print("rowDict:", rowDict)
    featureList.append(rowDict)

# print(labelList)
# print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

# print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

# print("labelList: " + str(labelList))
lb = LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print("dummyY: " + str(dummyY))


#生成决策树
#参照http://scikit-learn.org/stable/modules/tree.html
#默认是基于index cart算法
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
# print("clf: " + str(clf))


#保存决策树
with open("allElectronicInformationGainOri.dot", 'w') as f:
    # f = tree.export_graphviz(clf, out_file = f)
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

#预测
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX

newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY" + str(predictedY))