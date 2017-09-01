import numpy as np
from sklearn import datasets, svm
from SPDRankSVM import SPDRankSVM
from scipy import stats
from datetime import datetime

np.random.seed(int(datetime.now().strftime('%s')))

data = datasets.load_diabetes()
X, ys = data["data"], data["target"]
N_train = 300
X_train, X_test = X[:N_train], X[N_train:]
ys_train, ys_test = ys[:N_train], ys[N_train:]

svr = svm.LinearSVR()
svr.fit(X_train, ys_train)
ranksvm = SPDRankSVM(n_iteration=1000000)
ranksvm.fit(X_train, ys_train)

ys_svr_pred = svr.predict(X_test)
ys_ranksvm_pred = ranksvm.predict(X_test)

print("SVR")
print(stats.kendalltau(ys_test, ys_svr_pred))

print("RankSVM")
print(stats.kendalltau(ys_test, ys_ranksvm_pred))
