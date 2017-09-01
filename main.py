# -*- coding: utf-8 -*-
import SGDSVC
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:100]
    y = iris.target[:100]
    # ラベルは1, -1
    y = map(lambda l: 1 if l == 1 else -1, y)
    # スケーリング
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    clf = SGDSVC.SGDSVC(max_iter = 100000)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Confusion Matrix")
    print(metrics.confusion_matrix(y, y_pred))
    print(metrics.classification_report(y, y_pred))
