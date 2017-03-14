# -*- coding: utf-8 -*-
# @Time    : 2017/2/25 下午4:15
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import numpy as np
import random

"""
all data_set with '*.npz' are created by "row by row"
remember this.
"""

DATA_SET_PATH = 'data'


def get_triple_data(res_num, src_num, class_num, X, Y):
    """
    :param res_num: result data number
    :param src_num: srouce data number
    :param class_num: class number
    :param X: train input data set
    :param Y: train output data set
    :return: triplet data (just for train_X)
    """
    class2train_x = [[] for i in xrange(class_num)]

    for i in xrange(src_num):
        for j, y in enumerate(Y[i]):
            if y != 0.0:
                class2train_x[j].append(X[i])

    class2train_num = [len(class2train_x[i]) for i in xrange(class_num)]

    res_X = []

    for _ in xrange(res_num):
        # produce the class_num and allocation to Y
        x_class, x_minus_class = random.sample(range(class_num), 2)

        # produce the triplet
        x_plus_idx = x_idx = random.randint(0, class2train_num[x_class] - 1)
        x_minus_idx = random.randint(0, class2train_num[x_minus_class] - 1)
        while x_plus_idx == x_idx:
            x_plus_idx = random.randint(0, class2train_num[x_class] - 1)

        # get selected data
        x = class2train_x[x_class][x_idx]
        x_plus = class2train_x[x_class][x_plus_idx]
        x_minus = class2train_x[x_minus_class][x_minus_idx]

        # union triplet data
        res_X.append(np.hstack((x, x_plus, x_minus)))

    return np.array(res_X)


def get_contrastive_data(res_num, src_num, X, Y, shape=None):
    res_X = []
    res_Y = []
    for i in xrange(res_num):
        idx_1 = random.randint(0, src_num - 1)
        idx_2 = random.randint(0, src_num - 1)

        x1 = X[idx_1]
        y1 = Y[idx_1]
        x2 = X[idx_2]
        y2 = Y[idx_2]

        if shape:
            x1 = x1.reshape(shape)
            x2 = x2.reshape(shape)
            res_X.append([x1, x2])
        else:
            res_X.append(np.hstack((x1, x2)))

        res_Y.append((y1 == y2).all() * 1)

    return np.array(res_X), np.array(res_Y)


def create_triplet_data(npz_path, dest_path):
    src = np.load(npz_path)

    # source data
    src_train_X = src['train_X']
    src_train_Y = src['train_Y']
    src_test_X = src['test_X']
    src_test_Y = src['test_Y']

    # source parameter
    src_train_num = np.shape(src_train_X)[0]
    src_test_num = np.shape(src_test_X)[0]
    class_num = np.shape(src_train_Y)[1]

    # triple data parameter
    train_num = 70000
    test_num = 30000

    # create triple train data
    train_X = get_triple_data(train_num, src_train_num, class_num, src_train_X, src_train_Y)

    # create test data
    test_X, test_Y = get_contrastive_data(test_num, src_test_num, src_test_X, src_test_Y, shape=(28, 28))

    # save triplet data
    np.savez(dest_path, train_X=train_X, test_X=test_X, test_Y=test_Y)
    print 'ok'
    print 'train_X: %s\ntest_X: %s\ntest_Y: %s' % (np.shape(train_X), np.shape(test_X), np.shape(test_Y))


def create_contrastive_data(npz_path, dest_path):
    src = np.load(npz_path)

    # source data
    src_train_X = src['train_X']
    src_train_Y = src['train_Y']
    src_test_X = src['test_X']
    src_test_Y = src['test_Y']

    # source parameter
    src_train_num = np.shape(src_train_X)[0]
    src_test_num = np.shape(src_test_X)[0]
    class_num = np.shape(src_train_Y)[1]

    # triple data parameter
    train_num = 70000
    test_num = 30000

    # create contrastive train data
    train_X, train_Y = get_contrastive_data(train_num, src_test_num, src_test_X, src_test_Y)

    # create test data
    test_X, test_Y = get_contrastive_data(test_num, src_test_num, src_test_X, src_test_Y, shape=(28, 28))

    # save triplet data
    np.savez(dest_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
    print 'ok'
    print 'train_X: %s\ntrain_Y: %s\ntest_X: %s\ntest_Y: %s' % (np.shape(train_X), np.shape(train_Y), np.shape(test_X), np.shape(test_Y))


def main():
    # create_triplet_data(DATA_SET_PATH + '/mnist.npz', DATA_SET_PATH + '/triplet_mnist.npz')
    create_contrastive_data(DATA_SET_PATH + '/mnist.npz', DATA_SET_PATH + '/contrastive_mnist.npz')


if __name__ == '__main__':
    main()
