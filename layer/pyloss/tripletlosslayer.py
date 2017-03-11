# -*- coding: utf-8 -*-
# @Time    : 2017/3/10 下午2:03
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import caffe
import numpy as np
# from numpy import *
# import yaml


class TripletLayer(caffe.Layer):
    # global no_residual_list, margin

    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""

        # print 123
        assert np.shape(bottom[0].data) == np.shape(bottom[1].data)
        assert np.shape(bottom[0].data) == np.shape(bottom[2].data)

        # layer_params = yaml.load(self.param_str_)
        # self.margin = layer_params['margin']

        self.margin = 0.2

        self.a = 1
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        anchor_minibatch_db = []
        positive_minibatch_db = []
        negative_minibatch_db = []
        for i in range((bottom[0]).num):
            anchor_minibatch_db.append(bottom[0].data[i])

            positive_minibatch_db.append(bottom[1].data[i])

            negative_minibatch_db.append(bottom[2].data[i])

        loss = float(0)
        self.no_residual_list = []
        for i in range(((bottom[0]).num)):
            a = np.array(anchor_minibatch_db[i])
            p = np.array(positive_minibatch_db[i])
            n = np.array(negative_minibatch_db[i])
            a_p = a - p
            a_n = a - n
            ap = np.dot(a_p, a_p)
            an = np.dot(a_n, a_n)
            dist = (self.margin + ap - an)
            _loss = max(dist, 0.0)
            if i == 0:
                pass
                # print ('loss:' + ' ap:' + str(ap) + ' ' + 'an:' + str(an))
            if _loss == 0:
                self.no_residual_list.append(i)
            loss += _loss

        loss = (loss / (2 * (bottom[0]).num))
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        count = 0
        if propagate_down[0]:
            for i in range((bottom[0]).num):
                if not i in self.no_residual_list:
                    x_a = bottom[0].data[i]
                    x_p = bottom[1].data[i]
                    x_n = bottom[2].data[i]

                    # print x_a,x_p,x_n
                    bottom[0].diff[i] = self.a * ((x_n - x_p) / ((bottom[0]).num))
                    bottom[1].diff[i] = self.a * ((x_p - x_a) / ((bottom[0]).num))
                    bottom[2].diff[i] = self.a * ((x_a - x_n) / ((bottom[0]).num))

                    count += 1
                else:
                    bottom[0].diff[i] = np.zeros(np.shape(bottom[0].data)[1])
                    bottom[1].diff[i] = np.zeros(np.shape(bottom[0].data)[1])
                    bottom[2].diff[i] = np.zeros(np.shape(bottom[0].data)[1])

                    # print 'select gradient_loss:',bottom[0].diff[0][0]
                    # print shape(bottom[0].diff),shape(bottom[1].diff),shape(bottom[2].diff)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



class VerificationLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need two inputs and one sims to compute accuracy.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")

        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)

    def forward(self, bottom, top):
        # input data
        threshold = bottom[3].data
        sims = bottom[2].data
        diff = bottom[0].data - bottom[1].data
        dis = np.sqrt(np.sum(diff ** 2, axis=1))
        normal_dis = (dis - np.min(dis)) / (np.max(dis) - np.min(dis))
        pred = normal_dis <= threshold

        # comfusion matrix
        TP = np.sum(pred * sims)
        FP = np.sum(pred * (1 - sims))
        FN = np.sum((1 - pred) * sims)
        TN = np.sum((1 - pred) * (1 - sims))

        # print normal_dis
        # print 'pred:', (normal_dis <= threshold).astype(np.int)

        # compute accuracy
        top[0].data[...] = np.mean(pred == sims)

        # compute TPR
        top[1].data[...] = TP / (TP + FN)

        # compute FPR
        top[2].data[...] = FP / (TN + FP)

        # top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.
        # top[0].data[...] = np.sqrt(np.sum(self.diff ** 2, axis=1))

    def backward(self, top, propagate_down, bottom):
        # for i in range(2):
        #     if not propagate_down[i]:
        #         continue
        #     if i == 0:
        #         sign = 1
        #     else:
        #         sign = -1
        #     bottom[i].diff[...] = sign * self.diff / bottom[i].num
        raise Exception("No backward!!!!")