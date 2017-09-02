
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys, os
import pandas as pd
import time
import tempfile
from collections import defaultdict
from functools import reduce

from network_base import Network
from tensorflow.examples.tutorials.mnist import input_data
from vecgen_tf import orthoconv_filter

import tensorflow as tf

FLAGS = None

class deepnn(Network):
  def __init__(self, ortho_conv=False, nonlin='relu'):
    assert nonlin in ['relu', 'selu']
    if nonlin=='relu':
      self._nonlin = self._relu
    elif nonlin=='selu':
      self._nonlin = self._selu

    if ortho_conv:
      self._conv_weights_fn = self._ortho_weight_bias_variable
    else:
      self._conv_weights_fn = self._weight_bias_variable

  def forward(self, x):
    with tf.name_scope('reshape'):
      x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope('conv1'):
      W_conv1, b_conv1 = self._conv_weights_fn([7, 7, 1, 32])
    
    h_conv1 = self._nonlin(self._conv2d(x_image, W_conv1, b_conv1))
    h_pool1 = self._max_pool_2x2(h_conv1)

    with tf.variable_scope('conv2'):
      W_conv2, b_conv2 = self._conv_weights_fn([5, 5, 32, 64])
    
    h_conv2 = self._nonlin(self._conv2d(h_pool1, W_conv2, b_conv2))
    h_pool2 = self._max_pool_2x2(h_conv2)

    with tf.variable_scope('conv3'):
      W_conv3, b_conv3 = self._conv_weights_fn([5, 5, 64, 128])

    h_conv3 = self._nonlin(self._conv2d(h_pool2, W_conv3, b_conv3))
    h_pool3 = self._max_pool_2x2(h_conv3)

    with tf.variable_scope('fc1'):
      W_fc1, b_fc1 = self._weight_bias_variable([4 * 4 * 128, 512])

    h_pool3_flat = self._flatten(h_pool3)
    h_fc1 = self._nonlin(self._dense(h_pool3_flat, W_fc1, b_fc1))

    with tf.variable_scope('fc2'):
      W_fc2, b_fc2 = self._weight_bias_variable([512, 10])

    y_conv = self._dense(h_fc1, W_fc2, b_fc2)
    return y_conv

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-path', default='./tmp/',
                      type=str, help='Directory for storing input data')
  parser.add_argument('--result-path', default='result', type=str, 
                      help='Directory for storing training and eval logs')
  parser.add_argument('--ortho-conv', action='store_true', default=False,
                      help='use orthogonal convolution')
  parser.add_argument('--nonlin', choices=['relu', 'selu'], default='relu', 
                      type=str, help='nonlinearity to use (default %(default)s)')

  options = parser.parse_args()
  assert not(os.path.exists(options.result_path)), "result dir already exists!"
  result_path = options.result_path
  os.makedirs(result_path)

  mnist = input_data.read_data_sets(options.data_path, one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.float32, [None, 10])

  net = deepnn(ortho_conv=options.ortho_conv, nonlin=options.nonlin)
  y_conv = net.forward(x)

  cross_entropy_vector = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy_vector)
  optimizer = tf.train.AdamOptimizer(1e-4)
  train_step = optimizer.minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_metrics = defaultdict(list)
    eval_metrics = defaultdict(list)
    train_start_time = time.time()
    for i in range(1, 20001):
      batch = mnist.train.next_batch(50)
      _, train_cross_entropy, train_accuracy = sess.run([train_step, cross_entropy, accuracy], 
                                                feed_dict={x: batch[0], y: batch[1]})
      train_metrics['time_per_iter'].append((time.time() - train_start_time)/i)
      train_metrics['iteration'].append(i)
      train_metrics['cross_entropy'].append(train_cross_entropy)
      train_metrics['accuracy'].append(train_accuracy)

      if (i - 1) % 100 == 0:
        eval_start_time = time.time()
        eval_cross_entropy, eval_accuracy = sess.run([cross_entropy, accuracy], 
                                              feed_dict={x: mnist.test.images, y: mnist.test.labels})
        eval_metrics['time_per_iter'].append(time.time() - eval_start_time)
        eval_metrics['iteration'].append(i)
        eval_metrics['cross_entropy'].append(eval_cross_entropy)
        eval_metrics['accuracy'].append(eval_accuracy)
        print('step %d, eval accuracy %g' % (i, eval_accuracy))

        pd_train_metrics = pd.DataFrame(train_metrics)
        pd_eval_metrics = pd.DataFrame(eval_metrics)

        pd_train_metrics.to_csv(os.path.join(result_path, 'train_metrics.csv'))
        pd_eval_metrics.to_csv(os.path.join(result_path, 'eval_metrics.csv'))

if __name__ == '__main__':
  main()
