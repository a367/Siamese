# -*- coding: utf-8 -*-
# @Time    : 2017/4/8 下午5:03
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

# the margin between ap and an
MARGIN = 0.5

BATCH_SIZE = 200

IDENTITY_NUM = 5

MIN_IDENTITY_SIZE = 20

MAX_IDENTITY_SIZE = 40

BLOB_SHAPE = (BATCH_SIZE, 1, 28, 28)

DATA_PATH = 'data/mnist/src_mnist.npz'

MEAN_VALUE = 127.5

SCALE = 0.0078125