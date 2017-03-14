import caffe
import numpy as np


class EuclideanDistanceLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

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





        np.set_printoptions(threshold='nan')
        with open('test.txt', 'w') as f:
            f.write(str(np.hstack((dis.reshape(-1,1), sims.reshape(-1,1)))))

        raise Exception("Inputs must have the same dimension.")






        normal_dis = (dis - np.min(dis)) / (np.max(dis) - np.min(dis))
        pred = normal_dis <= threshold

        # confusion matrix
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
