# -*- coding: utf-8 -*-
# @Time    : 2017/2/11 上午7:47
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import numpy as np
import json
import tensorflow as tf
import sys


class SLDataSet:
    def __init__(self, data):
        """
        :param data:
         data structure is set row by row
         which have 'train_X', 'train_X_plus', 'train_X_minus', indices
        """
        self.train_X = data['train_X'].T.astype("float32")
        self.train_X_plus = data['train_X_plus'].T.astype("float32")
        self.train_X_minus = data['train_X_minus'].T.astype("float32")

        self.feature_size, self.train_num = np.shape(self.train_X)


def train_msrl(data_set, params):
    # data parameters
    feature_size = data_set.feature_size
    train_num = data_set.train_num

    # shared variables
    x = tf.Variable(data_set.train_X, trainable=False)
    x_plus = tf.Variable(data_set.train_X_plus, trainable=False)
    x_minus = tf.Variable(data_set.train_X_minus, trainable=False)

    # recurrent changed parameters
    u = tf.Variable(np.array([[1.0] * train_num]).astype("float32"), trainable=False)
    v = tf.Variable(np.array([[1.0] * feature_size for _ in xrange(feature_size)]).astype("float32"), trainable=False)

    # u = tf.placeholder("float", [1, train_num])
    # v = tf.placeholder("float", [feature_size, feature_size])

    # model weights
    M = tf.Variable(tf.random_uniform((feature_size, feature_size), minval=0, maxval=1, dtype="float32"))

    # similarity matrix with x_plus and x_minus
    simi_x_plus = tf.reduce_sum(tf.transpose(tf.matmul(x, M, True)) * x_plus, reduction_indices=0)
    simi_x_minus = tf.reduce_sum(tf.transpose(tf.matmul(x, M, True)) * x_minus, reduction_indices=0)

    # hinge loss
    hinge_loss = tf.reduce_mean(u * tf.maximum(0., params['margin'] - simi_x_plus + simi_x_minus))

    # capped-L1 regularization term
    capped_L1 = params['lambda'] * tf.reduce_sum(v * tf.abs(M))

    # final cost formulation
    cost = hinge_loss + capped_L1

    check_u23 = tf.cast((0 <= params['margin'] - simi_x_plus + simi_x_minus), "float32")
    check_u12 = tf.cast(params['margin'] - simi_x_plus + simi_x_minus <= params['epsilon'], "float32")
    update_u = tf.reshape(check_u12, (1, -1))
    update_v = tf.cast(tf.abs(M) <= params['mu'], "float32")

    assign_u = u.assign(update_u)
    assign_v = v.assign(update_v)

    # Gradient Descent
    # optimizer = tf.train.GradientDescentOptimizer(params['lr']).minimize(cost)
    optimizer = tf.train.AdamOptimizer(params['lr']).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)

        # Training cycle
        for epoch in xrange(params['ep']):
            sess.run([optimizer, cost])
            train_u = sess.run(assign_u)
            train_v = sess.run(assign_v)

            if epoch % 10 == 0:
                triplet_loss = sess.run(hinge_loss)
                norm_loss = sess.run(capped_L1)
                util_u12 = np.sum(train_u) / float(train_u.size)
                util_v = np.sum(train_v) / float(train_v.size)
                util_u23 = sess.run(tf.reduce_mean(check_u23))
                util_u1, util_u2, util_u3 = 1 - util_u23, util_u12 + util_u23 - 1, 1 - util_u12

                print '[%d / %d] loss:%s = %s + %s, u:%s, v:%s' % (epoch, params['ep'], triplet_loss + norm_loss, triplet_loss, norm_loss, util_u12, util_v)
                print 'simple part(unused): %s, semi part: %s, hard part(unused): %s' % (util_u1, util_u2, util_u3)
                print

        res = M.eval()
    tf.reset_default_graph()
    return res


def compute_accuracy(data_set, M):
    # res = np.dot(np.dot(data_set.test_X.T, M), data_set.train_X_total)
    # max_idxs = np.argmax(res, axis=1)  # shape is (1, m)
    # pred = data_set.train_Y[:, max_idxs].reshape(1, -1)
    # ac = np.mean(np.equal(pred, data_set.test_Y))
    #
    # return ac
    pass


def main(data_npz_path):
    pass
