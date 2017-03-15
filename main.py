# -*- coding: utf-8 -*-
# @Time    : 2017/3/7 下午7:43
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn


import os
import lmdb
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array
import matplotlib.pyplot as plt
from scipy.misc import toimage


def generate_mean_image():
    os.system('~/caffe/build/tools/compute_image_mean -backend=lmdb data/train_lmdb data/mean.binaryproto')


def get_data_for_case_from_lmdb(lmdb_name, idx):
    lmdb_env = lmdb.open(lmdb_name, readonly=True)
    lmdb_txn = lmdb_env.begin()

    raw_datum = lmdb_txn.get(idx)
    datum = caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    feature = datum_to_array(datum)
    label = datum.label

    return label, feature


def train(solver_path, log_path):
    os.system('~/caffe/build/tools/caffe train --solver %s -gpu 3  2>&1 | tee %s' % (solver_path, log_path))


def compute_accuracy(net, pair_imgs, sims, threshold):
    out = net.forward(pair_data=pair_imgs, sim=sims, threshold=threshold)

    return out['accuracy'][0], out['TPR'][0], out['FPR'][0]


def apply(deploy_path, npz_path):
    net = caffe.Net(deploy_path, "log/_iter_50000.caffemodel", caffe.TEST)
    # net = caffe.Net(deploy_path, "test.caffemodel", caffe.TEST)

    data = np.load(npz_path)

    test_X = []
    test_Y = []
    for i in xrange(10000):
        label, img = get_data_for_case_from_lmdb('data/contrastive_test_lmdb', '%.8d' % i)
        test_X.append(img)
        test_Y.append(label)

    # test_X = data['test_X'][:10000] / 256.0
    # test_Y = data['test_Y'][:10000]
    # net.blobs['pair_data'].reshape(*pair_imgs.shape)

    test_X = np.array(test_X) / 256.0
    test_Y = np.array(test_Y)

    # out = net.forward(pair_data=test_X, sim=test_Y, threshold=np.array([1]))
    #
    # print out['loss']
    # return

    TPR_arr = []
    FPR_arr = []

    point_num = 50

    for i in xrange(point_num):
        accuracy, TPR, FPR = compute_accuracy(net, test_X, test_Y, np.array([i / float(point_num)]))
        print 'accuracy: %s,   TPR: %s,   FPR: %s' % (accuracy, TPR, FPR)
        TPR_arr.append(TPR)
        FPR_arr.append(FPR)

    print TPR_arr
    print FPR_arr


if __name__ == '__main__':
    # train triplet loss
    # train('model/triplet_solver.prototxt', 'log/train.log')

    # train contrastive loss
    # train('model/siamese/mnist_siamese_solver.prototxt', 'log/siamese/train.log')

    apply("model/triplet_deploy.prototxt", "data/contrastive_mnist.npz")
