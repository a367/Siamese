# -*- coding: utf-8 -*-
# @Time    : 2017/2/20 下午4:22
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import tensorflow as tf
import numpy as np

feature_size = 4723
M = tf.constant(np.zeros([feature_size, feature_size], "float"))
m_add = M + 1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in xrange(1000):
        c = sess.run(m_add)
    print 'ok'

print '123'
