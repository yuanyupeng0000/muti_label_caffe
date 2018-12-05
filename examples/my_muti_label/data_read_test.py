#!/usr/bin/env python3
#! --*-- coding: utf-8 --*--
import numpy as np
import matplotlib.pyplot as plt

import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)
# caffe.set_mode_cpu()


print('Start...')
solver_def = 'train_multilabel_solver.prototxt'

solver = caffe.SGDSolver(solver_def)
solver.step(1)

data = solver.net.blobs['data'].data
labels = solver.net.blobs['label'].data

img = np.transpose(data[0], (1, 2, 0))
gt = labels[0]
print('ground_truth: {0}'.format(gt))
plt.imshow(img)
plt.show()

print('Done.')
