"""
SPDRankSVM: Stochastic Pairwise Descent RankSVM
"""

import numpy as np


class SPDRankSVM(object):

    def __init__(self, lmd=1e-5, n_iteration=100):
        self.lmd = lmd
        self.n_iteration = n_iteration

    def fit(self, X, ys):
        i =0
        N, D = X.shape
        self.w = np.zeros(D)

        for i_iteration in range(1, self.n_iteration + 1):
            first_ind = np.random.random_integers(0, N - 1)
            second_ind = np.random.random_integers(0, N - 1)
            pair_x = (X[first_ind], X[second_ind])
            pair_y = (ys[first_ind], ys[second_ind])
            if pair_y[0] > pair_y[1]:
                y_diff = 1
            elif pair_y[1] > pair_y[0]:
                y_diff = -1
            else:
                # 評価値が同じときは重みを更新しない
                i = i +1
                continue

            eta = 1. / (self.lmd * i_iteration)
            x_diff = pair_x[0] - pair_x[1]
            # y * <w, x> < 1  -> (1 - eta * lmd) * w + eta * y * x
            # y * <w, x> >= 1 -> (1 - eta * lmd) * w
            self.w *= (1 - eta * self.lmd)
            if y_diff * x_diff.dot(self.w) < 1:
                self.w += eta * y_diff * x_diff
        print(i)

    def predict(self, X):
        return X.dot(self.w)
