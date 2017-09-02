
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
import sys, os
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

  def cross_entropy_and_accuracy(labels, logits):
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

  options = parser.parse_args()
  if options.dataset=='cifar10':
    num_classes = 10
  elif options.dataset=='cifar100':
    num_classes = 100
  assert not(os.path.exists(options.result_path)), "result dir already exists!"
  result_path = options.result_path

  net = deepnn(ortho_conv=options.ortho_conv, nonlin=options.nonlin)

  train_images, train_labels = cifar_input.build_input(options.dataset, 128, 'train')
  eval_images, eval_labels = cifar_input.build_input(options.dataset, 100, 'eval')

  with tf.variable_scope('network'):
    train_logits = net.forward(train_images, num_classes)
    train_cross_entropy, train_accuracy = net.cross_entropy_and_accuracy(train_labels, train_logits)

  with tf.variable_scope('network', reuse=True):
    eval_logits = net.forward(eval_images, num_classes)
    eval_cross_entropy_vector = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

  optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
  train_step = optimizer.minimize(train_cross_entropy)

  with tf.Session() as sess:
    train_metrics = defaultdict(list)
    eval_metrics = defaultdict(list)
    os.makedirs(result_path)

    sess.run(tf.global_variables_initializer())
    for i in range(1, 20001):
      _, train_cross_entropy, train_accuracy = sess.run([train_step, train_cross_entropy, train_accuracy])
      train_metrics['iteration'].append(i)
      train_metrics['cross_entropy'].append(train_cross_entropy)
      train_metrics['accuracy'].append(train_accuracy)

      if (i - 1) % 100 == 0:
        total_prediction, correct_prediction = 0, 0
        total_eval_cross_entropy = 0
        for _ in range(EVAL_BATCH_COUNT):
          eval_cross_entropy_vector_val, eval_predictions_val, eval_truth_val = sess.run([eval_cross_entropy_vector, 
                                                                                  eval_logits, eval_labels])
          total_eval_cross_entropy += np.sum(eval_cross_entropy_vector_val)
          truth = np.argmax(eval_truth_val, axis=1)
          predictions = np.argmax(eval_predictions_val, axis=1)
          correct_prediction += np.sum(truth == predictions)
          total_prediction += predictions.shape[0]
        eval_avg_precision = correct_prediction / float(total_prediction)
        eval_avg_cross_entropy = total_eval_cross_entropy / float(EVAL_BATCH_COUNT)

        eval_metrics['iteration'].append(i)
        eval_metrics['cross_entropy'].append(eval_avg_cross_entropy)
        eval_metrics['accuracy'].append(eval_avg_precision)
        print('step %d, eval accuracy %g, eval loss' % (i, eval_accuracy, eval_cross_entropy))

        pd_train_metrics = pd.DataFrame(train_metrics)
        pd_eval_metrics = pd.DataFrame(eval_metrics)

        pd_train_metrics.to_csv(os.path.join(result_path, 'train_metrics.csv'))
        pd_eval_metrics.to_csv(os.path.join(result_path, 'eval_metrics.csv'))


if __name__ == '__main__':
  main()