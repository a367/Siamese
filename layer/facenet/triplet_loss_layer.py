# -*- coding: utf-8 -*-
# @Time    : 2017/4/7 下午10:31
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import caffe
import numpy as np
import config


class TripletLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.margin = config.MARGIN
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        x = bottom[0].data
        x_p = bottom[1].data
        x_m = bottom[2].data

        dis_p = np.sum(np.square((x - x_p)), axis=1)
        dis_m = np.sum(np.square((x - x_m)), axis=1)

        loss = dis_p - dis_m + self.margin
        self.check = (loss >= 0).reshape(-1, 1)
        top[0].data[...] = np.mean(np.maximum(0, loss))

    def backward(self, top, propagate_down, bottom):
        x = bottom[0].data
        x_p = bottom[1].data
        x_m = bottom[2].data

        bottom[0].diff[...] = 2.0 * (x_m - x_p) * self.check / bottom[0].num
        bottom[1].diff[...] = 2.0 * (x_p - x) * self.check / bottom[1].num
        bottom[2].diff[...] = 2.0 * (x - x_m) * self.check / bottom[1].num


class TripletLossLayer2(caffe.Layer):
    def setup(self, bottom, top):
        self.margin = config.MARGIN
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        x = bottom[0].data
        x_p = bottom[1].data
        x_m = bottom[2].data

        dis_p = np.sum(np.square((x - x_p)), axis=1)
        self.dis_m = np.sqrt(np.sum(np.square((x - x_m)), axis=1))
        loss_m = np.square(np.maximum(0, self.margin - self.dis_m))
        self.check = ((self.margin - self.dis_m) >= 0).reshape(-1, 1)

        top[0].data[...] = np.mean(loss_m + dis_p) / 2.0

    def backward(self, top, propagate_down, bottom):
        x = bottom[0].data
        x_p = bottom[1].data
        x_m = bottom[2].data

        bottom[1].diff[...] = (x_p - x) / bottom[1].num
        bottom[2].diff[...] = ((self.dis_m - self.margin) / (1e-4 + self.dis_m)).reshape(-1, 1) * (x_m - x) * self.check / bottom[2].num
        bottom[0].diff[...] = -bottom[1].diff - bottom[2].diff
