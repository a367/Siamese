# -*- coding: utf-8 -*-
# @Time    : 2017/3/7 下午7:43
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn


import os
import lmdb
import numpy as np
# import cv2, cv
import caffe
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array
import matplotlib.pyplot as plt
from scipy.misc import toimage


def get_data_for_case_from_lmdb(lmdb_name, idx):
    lmdb_env = lmdb.open(lmdb_name, readonly=True)
    lmdb_txn = lmdb_env.begin()

    raw_datum = lmdb_txn.get(idx)
    datum = caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    feature = datum_to_array(datum)
    label = datum.label

    return label, feature


def main():
    solver_path = 'model/triplet_solver.prototxt'
    log_path = 'log/train.log'
    os.system('~/caffe/build/tools/caffe train --solver %s -gpu 1  2>&1 | tee %s' % (solver_path, log_path))

    # caffe.set_mode_gpu()
    # solver = caffe.get_solver(solver_path)
    # solver.solve()


def generate_mean_image():
    os.system('~/caffe/build/tools/compute_image_mean -backend=lmdb data/train_lmdb data/mean.binaryproto')


def compute_accuracy(net, pair_imgs, sims, threshold):
    out = net.forward(pair_data=pair_imgs, sim=sims, threshold=threshold)

    return out['accuracy'][0], out['TPR'][0], out['FPR'][0]


def apply():
    net = caffe.Net("model/siamese_deploy.prototxt", "model/_iter_50000.caffemodel", caffe.TEST)

    pair_imgs = []
    sims = []

    for i in xrange(10000):
        sim, pair_img = get_data_for_case_from_lmdb("data/test_lmdb/", "%.8d" % i)
        sims.append(sim)
        pair_imgs.append(pair_img)

    pair_imgs = np.array(pair_imgs)
    sims = np.array(sims)
    # net.blobs['pair_data'].reshape(*pair_imgs.shape)

    TPR_arr = []
    FPR_arr = []

    for i in xrange(200):
        accuracy, TPR, FPR = compute_accuracy(net, pair_imgs, sims, np.array([i / 200.0]))
        print 'accuracy: %s,   TPR: %s,   FPR: %s' % (accuracy, TPR, FPR)
        TPR_arr.append(TPR)
        FPR_arr.append(FPR)

    print TPR_arr
    print FPR_arr


if __name__ == '__main__':
    main()
    # apply()
