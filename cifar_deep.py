
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

"""A deep CIFAR-10 classifier using convolutional layers.
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
import pandas as pd
import sys, os
import time
import tempfile
from collections import defaultdict
from network_base import Network
import cifar_input

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

EVAL_BATCH_COUNT = 100

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

  def forward(self, x, num_classes):
    images_batch = tf.reshape(x, [-1, 32, 32, 3])

    with tf.variable_scope('conv1'):
      W_conv, b_conv = self._conv_weights_fn([5, 5, 3, 32])
    h = self._nonlin(self._conv2d(images_batch, W_conv, b_conv))

    with tf.variable_scope('conv2'):
      W_conv, b_conv = self._conv_weights_fn([3, 3, 32, 32])
    h = self._nonlin(self._conv2d(h, W_conv, b_conv, padding='VALID'))

    h = self._max_pool_2x2(h)

    with tf.variable_scope('conv3'):
      W_conv, b_conv = self._conv_weights_fn([3, 3, 32, 64])
    h = self._nonlin(self._conv2d(h, W_conv, b_conv))

    with tf.variable_scope('conv4'):
      W, b = self._conv_weights_fn([3, 3, 64, 64])
    h = self._nonlin(self._conv2d(h, W, b, padding='VALID'))

    h = self._max_pool_2x2(h, padding='VALID')

    h = self._flatten(h)

    with tf.variable_scope('fc1'):
      W, b = self._weight_bias_variable([6 * 6 * 64, 512])
    h = self._nonlin(self._dense(h, W, b))

    with tf.variable_scope('fc2'):
      W, b = self._weight_bias_variable([512, num_classes])
    logits = self._dense(h, W, b)
    return logits

  def cross_entropy_and_accuracy(self, labels, logits):
    cross_entropy_vector = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy = tf.reduce_mean(cross_entropy_vector)

    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    return cross_entropy, accuracy


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10', 
                      type=str, help='Dataset to train the model on (default %(default)s)')
  parser.add_argument('--result-path', default='result', type=str, 
                      help='Directory for storing training and eval logs')
  parser.add_argument('--ortho-conv', action='store_true', default=False,
                      help='use orthogonal convolution')
  parser.add_argument('--nonlin', choices=['relu', 'selu'], default='relu', 
                      type=str, help='nonlinearity to use (default %(default)s)')
  parser.add_argument('--num-epochs', default=50, type=int, 
                      help='number of epochs (default %(default)s)')

  options = parser.parse_args()
  if options.dataset=='cifar10':
    (x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()
    num_classes = 10
  elif options.dataset=='cifar100':
    (x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.cifar100.load_data()
    num_classes = 100
  else:
    raise ValueError('Invalid dataset name')

  x_train = x_train/255.
  y_train = tf.contrib.keras.utils.to_categorical(y_train, num_classes)
  x_test = x_test/255.
  y_test = tf.contrib.keras.utils.to_categorical(y_test, num_classes)

  assert not(os.path.exists(options.result_path)), "result dir already exists!"
  result_path = options.result_path

  net = deepnn(ortho_conv=options.ortho_conv, nonlin=options.nonlin)
  images = tf.placeholder(tf.float32, [None, 32, 32, 3])
  labels = tf.placeholder(tf.float32, [None, num_classes])

  logits = net.forward(images, num_classes)
  cross_entropy, accuracy = net.cross_entropy_and_accuracy(labels, logits)

  optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
  train_step = optimizer.minimize(cross_entropy)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    train_metrics = defaultdict(list)
    eval_metrics = defaultdict(list)
    os.makedirs(result_path)
    models_path = os.path.join(result_path, 'models')

    sess.run(tf.global_variables_initializer())
    train_start_time = time.time()
    iters = 0
    for i in range(options.num_epochs):
      for j in range(0, len(x_train), 128):
        iters = iters + 1
        _, train_cross_entropy, train_accuracy = sess.run([train_step, cross_entropy, accuracy], 
                                                   feed_dict={images:x_train[j:j+128], labels:y_train[j:j+128]})
        train_metrics['time_per_iter'].append((time.time() - train_start_time)/iters)
        train_metrics['iteration'].append(iters)
        train_metrics['cross_entropy'].append(train_cross_entropy)
        train_metrics['accuracy'].append(train_accuracy)

        if (iters-1) % 100 == 0:
          eval_start_time = time.time()
          eval_cross_entropy, eval_accuracy = sess.run([cross_entropy, accuracy], feed_dict={images:x_test, labels:y_test})
          eval_metrics['time_per_iter'].append(time.time() - eval_start_time)
          eval_metrics['iteration'].append(iters)
          eval_metrics['cross_entropy'].append(eval_cross_entropy)
          eval_metrics['accuracy'].append(eval_accuracy)
          print('step %d, train accuracy %g, train loss %g' % (iters, train_accuracy, train_cross_entropy))
          print('eval accuracy %g, eval loss %g' % (eval_accuracy, eval_cross_entropy))

          saver = tf.train.Saver()
          model_name = 'iter_{}'.format(iters)
          model_filepath = os.path.join(models_path, model_name)
          saver.save(sess, model_filepath, write_meta_graph=False, write_state=False)

          pd_train_metrics = pd.DataFrame(train_metrics)
          pd_eval_metrics = pd.DataFrame(eval_metrics)

          pd_train_metrics.to_csv(os.path.join(result_path, 'train_metrics.csv'))
          pd_eval_metrics.to_csv(os.path.join(result_path, 'eval_metrics.csv'))

if __name__ == '__main__':
  main()
