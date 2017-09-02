from vecgen_tf import orthoconv_filter
from functools import reduce
import tensorflow as tf

class Network(object):
  def _conv2d(self, x, weight, bias, padding='SAME'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding=padding) + bias

  def _max_pool_2x2(self, x, padding='SAME'):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding=padding)

  def _flatten(self, x):
    shape = x.get_shape().as_list()
    shape_product = reduce((lambda x, y: x * y), shape[1:])
    return tf.reshape(x, [-1, shape_product])

  def _dense(self, x, weight, bias):
    return tf.matmul(x, weight) + bias

  def _relu(self, x):
    return tf.nn.relu(x)

  def _selu(self, x):
    """selu, self normalizing activation function"""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(tf.less(x, 0.0), alpha * tf.nn.elu(x), x)

  def _ortho_weight_bias_variable(self, shape):
    """generates an orthogonal weight and bias variable of a given shape."""
    assert len(shape)==4
    [in_height, in_width, in_channels, out_channels] = shape
    weight, bias = orthoconv_filter(in_height, in_width, in_channels, out_channels, bias=True)
    return weight, bias

  def _weight_bias_variable(self, shape):
    """generates a weight and bias variable of a given shape."""
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
    return weight, bias
