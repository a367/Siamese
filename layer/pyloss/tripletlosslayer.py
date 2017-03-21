# -*- coding: utf-8 -*-
# @Time    : 2017/3/10 下午2:03
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import caffe
import numpy as np


class Norm2Layer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        x = bottom[0].data
        x_square_sum = np.sum(np.square(x), axis=1).reshape(-1, 1)
        top[0].data[...] = x / np.sqrt(x_square_sum)

    def backward(self, top, propagate_down, bottom):
        x = bottom[0].data
        y = top[0].data

        x_square_sum = np.sum(np.square(x), axis=1).reshape(-1, 1)
        grad = top[0].diff - y * np.sum(top[0].diff * y, axis=1).reshape(-1, 1)

        bottom[0].diff[...] = grad / np.sqrt(x_square_sum)


class TripletLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.margin = 1
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


class TripletLayer2(caffe.Layer):
    def setup(self, bottom, top):
        self.margin = 1
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


class ContrastiveLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.margin = 1
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        x = bottom[0].data
        x_2 = bottom[1].data
        sim = bottom[2].data

        # legacy version
        dis_p = np.sum(np.square((x - x_2)), axis=1)
        loss_m = np.maximum(0, self.margin - dis_p)
        self.check = (self.margin - dis_p >= 0).reshape(-1, 1)

        # current version
        # dis_p = np.sum(np.square((x - x_2)), axis=1)
        # self.dis_m = np.sqrt(dis_p)
        # loss_m = np.square(np.maximum(0, self.margin - self.dis_m))
        # self.check = (self.margin - self.dis_m >= 0).reshape(-1, 1)

        # loss
        top[0].data[...] = np.mean((1 - sim) * loss_m + sim * dis_p) / 2.0

    def backward(self, top, propagate_down, bottom):
        x = bottom[0].data
        x_2 = bottom[1].data
        sim = bottom[2].data.reshape(-1, 1)

        alpha = top[0].diff / float(bottom[0].num)

        # legacy version
        grad_sim = (x - x_2) * alpha
        grad_not_sim = (x_2 - x) * self.check * alpha

        # current version
        # grad_sim = (x - x_2) * alpha
        # grad_not_sim = ((self.dis_m - self.margin) / (self.dis_m + 1e-4)).reshape(-1, 1) * (x - x_2) * self.check * alpha

        grad = grad_sim * sim + grad_not_sim * (1 - sim)
        bottom[0].diff[...] = grad
        bottom[1].diff[...] = -grad
