# -*- coding: utf-8 -*-
# @Time    : 2017/2/11 上午7:47
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import numpy as np
import json
import random
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


def train_mrsl(data_set, params):
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


def train_sl(data_set, params):
    # data parameters
    feature_size = data_set.feature_size
    train_num = data_set.train_num

    # shared variables
    x = tf.Variable(data_set.train_X, trainable=False)
    x_plus = tf.Variable(data_set.train_X_plus, trainable=False)
    x_minus = tf.Variable(data_set.train_X_minus, trainable=False)

    # model weights
    M = tf.Variable(tf.random_uniform((feature_size, feature_size), minval=0, maxval=1, dtype="float32"))

    # similarity matrix with x_plus and x_minus
    simi_x_plus = tf.reduce_sum(tf.transpose(tf.matmul(x, M, True)) * x_plus, reduction_indices=0)
    simi_x_minus = tf.reduce_sum(tf.transpose(tf.matmul(x, M, True)) * x_minus, reduction_indices=0)

    # similarity loss(or called triplet loss)
    sim_loss = params['margin'] - simi_x_plus + simi_x_minus

    simple_part = tf.reduce_mean(tf.cast(sim_loss < 0, dtype="float32"))
    hard_part = tf.reduce_mean(tf.cast(sim_loss > params['epsilon'], dtype="float32"))

    # hinge loss
    hinge_loss = tf.reduce_mean(tf.minimum(tf.maximum(0., sim_loss), params['epsilon']))

    # capped-L1 regularization term
    capped_L1 = params['lambda'] * tf.reduce_sum(tf.minimum(tf.abs(M), params['mu']))

    # final cost formulation
    cost = hinge_loss + capped_L1

    # Gradient Descent
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

            if epoch % 10 == 0:
                triplet_loss = sess.run(hinge_loss)
                norm_loss = sess.run(capped_L1)
                util_1 = sess.run(simple_part)
                util_3 = sess.run(hard_part)
                util_2 = 1 - util_1 - util_3

                print '[%d / %d] loss:%s = %s + %s' % (epoch, params['ep'], triplet_loss + norm_loss, triplet_loss, norm_loss)
                print 'simple part(unused): %s, semi part: %s, hard part(unused): %s' % (util_1, util_2, util_3)
                print

        res = M.eval()
    tf.reset_default_graph()
    return res


def train_sl_with_batch(data_set, params, batch_size):
    # data parameters
    feature_size = data_set.feature_size
    train_num = data_set.train_num
    batch_num = train_num / batch_size

    x = tf.placeholder(tf.float32, [feature_size, None])
    x_plus = tf.placeholder(tf.float32, [feature_size, None])
    x_minus = tf.placeholder(tf.float32, [feature_size, None])

    # model weights
    M = tf.Variable(tf.random_uniform((feature_size, feature_size), minval=0, maxval=1, dtype="float32"))

    # similarity matrix with x_plus and x_minus
    simi_x_plus = tf.reduce_sum(tf.transpose(tf.matmul(x, M, True)) * x_plus, reduction_indices=0)
    simi_x_minus = tf.reduce_sum(tf.transpose(tf.matmul(x, M, True)) * x_minus, reduction_indices=0)

    # similarity loss(or called triplet loss)
    sim_loss = params['margin'] - simi_x_plus + simi_x_minus

    simple_part = tf.reduce_mean(tf.cast(sim_loss < 0, dtype="float32"))
    hard_part = tf.reduce_mean(tf.cast(sim_loss > params['epsilon'], dtype="float32"))

    # hinge loss
    hinge_loss = tf.reduce_mean(tf.minimum(tf.maximum(0., sim_loss), params['epsilon']))

    # capped-L1 regularization term
    capped_L1 = params['lambda'] * tf.reduce_sum(tf.minimum(tf.abs(M), params['mu']))

    # final cost formulation
    cost = hinge_loss + capped_L1

    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(params['lr']).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        all_idx = range(train_num)

        # Training cycle
        for epoch in xrange(params['ep']):
            for i in xrange(batch_num - 1):
                f_dict = {x: data_set.train_X[:, i * batch_size:(i + 1) * batch_size],
                          x_plus: data_set.train_X_plus[:, i * batch_size:(i + 1) * batch_size],
                          x_minus: data_set.train_X_minus[:, i * batch_size:(i + 1) * batch_size]}

                sess.run([optimizer, cost], feed_dict=f_dict)

            random_idx = random.sample(all_idx, batch_size)
            f_dict = {x: data_set.train_X[:, random_idx], x_plus: data_set.train_X_plus[:, random_idx], x_minus: data_set.train_X_minus[:, random_idx]}

            triplet_loss = sess.run(hinge_loss, feed_dict=f_dict)
            norm_loss = sess.run(capped_L1)
            util_1 = sess.run(simple_part, feed_dict=f_dict)
            util_3 = sess.run(hard_part, feed_dict=f_dict)
            util_2 = 1 - util_1 - util_3
            #
            print '[%d / %d] loss:%s = %s + %s' % (epoch, params['ep'], triplet_loss + norm_loss, triplet_loss, norm_loss)
            print 'simple part(unused): %s, semi part: %s, hard part(unused): %s' % (util_1, util_2, util_3)
            print

        res = M.eval()
    tf.reset_default_graph()
    return res
