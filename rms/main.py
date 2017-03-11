# -*- coding: utf-8 -*-
# @Time    : 2017/2/11 上午7:47
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import numpy as np
import json
import tensorflow as tf
import sys


class SLDataSet:
    def __init__(self, npz_path):
        res = np.load(npz_path)

        self.train_X = res['train_X'].astype("float32")
        self.train_X_plus = res['train_X_plus'].astype("float32")
        self.train_X_minus = res['train_X_minus'].astype("float32")
        self.train_X_total = np.hstack((self.train_X, self.train_X_plus, self.train_X_minus))
        self.train_Y = res['train_Y']
        self.test_X = res['test_X']
        self.test_Y = res['test_Y']

        self.feature_size, self.train_num = np.shape(self.train_X)


def train_msrl(data_set, params):
    # data parameters
    feature_size = data_set.feature_size
    train_num = data_set.train_num

    # shared veriables
    x = tf.Variable(data_set.train_X, trainable=False)
    x_plus = tf.Variable(data_set.train_X_plus, trainable=False)
    x_minus = tf.Variable(data_set.train_X_minus, trainable=False)

    # recurrent changed parameters
    u = tf.placeholder("float", [1, train_num])
    v = tf.placeholder("float", [feature_size, feature_size])

    # model weights
    M = tf.Variable(tf.zeros([feature_size, feature_size], "float"))

    # similarity matrix with x_plus and x_minus
    simi_x_plus = tf.reduce_sum(tf.transpose(tf.matmul(x, M, True)) * x_plus, reduction_indices=0)
    simi_x_minus = tf.reduce_sum(tf.transpose(tf.matmul(x, M, True)) * x_minus, reduction_indices=0)

    # hinge loss
    hinge_loss = tf.reduce_mean(u * tf.maximum(0., 1 - simi_x_plus + simi_x_minus))

    # capped-L1 regularization term
    capped_L1 = params['lambda'] * tf.reduce_sum(v * tf.abs(M))

    # final cost formulation
    cost = hinge_loss + capped_L1

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(params['lr']).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)

        train_u = [[1.0] * train_num]
        train_v = [[1.0] * feature_size for _ in xrange(feature_size)]

        # Training cycle
        for epoch in range(params['ep']):
            _, c = sess.run([optimizer, cost], feed_dict={u: train_u, v: train_v})
            train_u = sess.run(tf.cast(simi_x_plus - simi_x_minus >= 1 - params['epsilon'], "float")).reshape(1, -1)
            train_v = sess.run(tf.cast(tf.abs(M) <= params['mu'], "float"))

        res = M.eval()

    tf.reset_default_graph()
    return res


def compute_accuracy(data_set, M):
    res = np.dot(np.dot(data_set.test_X.T, M), data_set.train_X_total)
    max_idxs = np.argmax(res, axis=1) # shape is (1, m)
    pred = data_set.train_Y[:, max_idxs].reshape(1, -1)
    ac = np.mean(np.equal(pred, data_set.test_Y))

    return ac



def main(data_npz_path):
    data_set = SLDataSet(data_npz_path)

    # hyper parameters set
    params = {'lr': 0.1, 'ep': 100, 'lambda': 0.01, 'epsilon': 10, 'mu': 0.001}

    M = train_msrl(data_set, params)
    ac = compute_accuracy(data_set, M)

    print ac


if __name__ == '__main__':
    main('data_set/sl_mnist.npz')
