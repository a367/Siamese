# -*- coding: utf-8 -*-
# @Time    : 2017/4/7 下午3:05
# @Author  : Zhixin Piao 
# @Email   : piaozhx@seu.edu.cn

import caffe
import numpy as np
import config
from collections import defaultdict


class TripletSelectLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the TripletSampleLayer."""
        top[0].reshape(*bottom[0].data.shape)
        top[1].reshape(*bottom[0].data.shape)
        top[2].reshape(*bottom[0].data.shape)
        self.margin = config.MARGIN

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""

        def get_semi_hard_negative_case():
            for label, idxs in label2idx.iteritems():
                if label == a_label:
                    continue

                for idx in idxs:
                    n_data = bottom_data[idx]
                    ap = np.sum((a_data - p_data) ** 2)
                    an = np.sum((a_data - n_data) ** 2)
                    if ap <= an <= ap + self.margin:
                        return n_data, idx
            return None, None

        bottom_data = np.array(bottom[0].data)
        bottom_label = np.array(bottom[1].data)
        self.idx_map = []

        top_a = []
        top_p = []
        top_n = []

        # key: label, value: [idx1, idx2, ...]
        label2idx = defaultdict(list)
        for i in xrange(bottom[0].num):
            label2idx[bottom_label[i]].append(i)

        for a_idx in xrange(bottom[0].num):
            a_label = bottom_label[a_idx]
            a_data = bottom_data[a_idx]

            for p_idx in label2idx[a_label]:
                p_data = bottom_data[p_idx]
                n_data, n_idx = get_semi_hard_negative_case()

                if p_idx == a_idx or n_data is None:
                    continue

                top_a.append(a_data)
                top_p.append(p_data)
                top_n.append(n_data)
                self.idx_map.append([a_idx, p_idx, n_idx])

        # print 'total number of triplet is %d' % len(self.idx_map)

        top[0].reshape(*np.array(top_a).shape)
        top[1].reshape(*np.array(top_a).shape)
        top[2].reshape(*np.array(top_a).shape)
        top[0].data[...] = np.array(top_a)
        top[1].data[...] = np.array(top_p)
        top[2].data[...] = np.array(top_n)

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""

        if propagate_down[0]:
            bottom[0].diff[...] = np.zeros(bottom[0].diff.shape)

            for i in xrange(top[0].num):
                bottom[0].diff[self.idx_map[i][0]] += top[0].diff[i]
                bottom[0].diff[self.idx_map[i][1]] += top[1].diff[i]
                bottom[0].diff[self.idx_map[i][2]] += top[2].diff[i]

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
