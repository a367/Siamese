# -*- coding: utf-8 -*-
# @Time    : 2017/2/10 上午6:25
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import tensorflow as tf
import random
import json
import numpy as np
import sys

# res = []
# for i in xrange(3):
#     with open('result_%d.json'%i, 'r') as f:
#         res += json.load(f)
#
# res = sorted(res,key=lambda x:-x[5])
# for datas in res:
#     for data in datas:
#         print data,
#     print

# A = np.array([[1,0,1],[0,1,0],[1,0,0]])
#
# A = np.array([[1,3,6,7,2,5,7,8]])
# B = np.array([[1,3,3,7]])
# print np.shape(A), np.shape(B)
# N = A[:,B].reshape(1,-1)
# print N
# print np.shape(N)



A = np.equal([0, 1, 3], np.arange(3))
print np.mean(A)



# aa = 2
# m = tf.Variable([[1, 2, 3]])
# xxx = m*aa
#
# def train():
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print sess.run(xxx)
#         sess.run(m.assign([[2, 4, 6]]))
#
#
# train()
# aa = 3
# train()



# cost = x * u * m
# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     # c = sess.run(cost, feed_dict={x: [[1,2],[3,4]], u: [[1,2],[3,4]]})
#
#     _, c = sess.run([optimizer, cost], feed_dict={x: [[1, 2], [3, 4]],
#                                                   u: [[1, 2], [3, 4]]})
#
#     print m.eval()
#     zz = tf.cast(m >= 0.9, "float")
#     print zz
#
#     # print sess.run(cost, feed_dict={x: [[1,2],[3,4]], })
