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
    #############
    # load data #
    #############
    class2train_x = [[] for i in xrange(class_num)]

    for i in xrange(src_num):
        class2train_x[Y[i]].append(X[i])

    class2train_num = [len(class2train_x[i]) for i in xrange(class_num)]

    # "multi" means train_num in this identity >= 2
    multi_train_set = filter(lambda x: x != None, [i if class2train_num[i] >= 2 else None for i in xrange(class_num)])

    ########################################
    # train data, shape=(N, 3 * row * col) #
    ########################################
    res_X = []
    for _ in xrange(res_num):
        # produce the class_num and allocation to Y
        x_minus_class = x_class = random.sample(multi_train_set, 1)[0]
        while x_minus_class == x_class:
            x_minus_class = random.randint(0, class_num - 1)

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


def get_contrastive_data(res_num, src_num, class_num, X, Y, shape=None):
    #############
    # load data #
    #############
    class2test_x = [[] for i in xrange(class_num)]

    for i in xrange(src_num):
        class2test_x[Y[i]].append(X[i])

    class2test_num = [len(class2test_x[i]) for i in xrange(class_num)]

    # "multi" means train_num in this identity >= 2
    multi_train_set = filter(lambda x: x != None, [i if class2test_num[i] >= 2 else None for i in xrange(class_num)])

    #####################################
    # test data, shape=(N, 2, row, col) #
    #####################################
    res_X = []
    res_Y = []
    for _ in xrange(res_num):
        sim = random.randint(0, 1)
        if sim == 0:
            x1_class, x2_class = random.sample(range(class_num), 2)
        else:
            x1_class = x2_class = random.sample(multi_train_set, 1)[0]

        idx_1 = random.randint(0, class2test_num[x1_class] - 1)
        idx_2 = random.randint(0, class2test_num[x2_class] - 1)
        while sim == 1 and idx_1 == idx_2:
            idx_2 = random.randint(0, class2test_num[x2_class] - 1)

        x1 = class2test_x[x1_class][idx_1].reshape(shape)
        x2 = class2test_x[x2_class][idx_2].reshape(shape)

        res_X.append([x1, x2])
        res_Y.append(sim)

    return np.array(res_X).astype(np.uint8), np.array(res_Y).astype(np.uint8)


def create_triplet_data(npz_path, dest_path):
    src = np.load(npz_path)

    # source data
    # test_X, test_X and train_X train_Y use same data set
    # this possibly cause some problem, but I'm not sure about it.
    src_train_X = src['train_X']
    src_train_Y = src['train_Y']
    src_test_X = src['train_X']
    src_test_Y = src['train_Y']

    print src_test_X.shape

    # source parameter
    src_train_num = np.shape(src_train_X)[0]
    src_test_num = np.shape(src_train_X)[0]
    class_num = 5749

    # triple data parameter
    train_num = 14000
    test_num = 6000

    # create triple train data
    train_X = get_triple_data(train_num, src_train_num, class_num, src_train_X, src_train_Y)

    # create test data
    test_X, test_Y = get_contrastive_data(test_num, src_test_num, class_num, src_test_X, src_test_Y, shape=(3, 64, 64))

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
    train_num = src_train_num
    test_num = src_test_num

    # create contrastive train data
    train_X, train_Y = get_contrastive_data(train_num, src_train_num, src_train_X, src_train_Y)

    # create test data
    test_X, test_Y = get_contrastive_data(test_num, src_test_num, src_test_X, src_test_Y, shape=(28, 28))

    # save triplet data
    np.savez(dest_path, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
    print 'ok'
    print 'train_X: %s\ntrain_Y: %s\ntest_X: %s\ntest_Y: %s' % (np.shape(train_X), np.shape(train_Y), np.shape(test_X), np.shape(test_Y))


def main():
    create_triplet_data(DATA_SET_PATH + '/lfw.npz', DATA_SET_PATH + '/triplet_lfw.npz')
    # create_contrastive_data(DATA_SET_PATH + '/mnist.npz', DATA_SET_PATH + '/contrastive_mnist.npz')


if __name__ == '__main__':
    main()
