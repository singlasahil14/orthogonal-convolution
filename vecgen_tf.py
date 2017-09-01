import numpy as np
import time
import tensorflow as tf

def check(w,b):
    w = np.reshape(w, [-1, w.shape[-1]])
    e = np.concatenate([w,b], axis=0)
    print(np.linalg.norm(e, axis=0))
    print(e.shape)
    min_dim = min(np.shape(e))
    ce = np.eye(min_dim, min_dim, dtype=np.float64)
    re = np.matmul(np.transpose(e), e)
    print(np.linalg.norm(ce-re))

def orthoconv_filter(in_height, in_width, in_channels, out_channels, 
                     num_ortho=None, bias=False, mode='train'):
    if(mode=='eval'):
        conv_filter = tf.get_variable(name='conv_filter') 
        conv_bias = tf.get_variable(name='conv_bias')
        return conv_filter, conv_bias
    m = in_height * in_width * in_channels
    n = out_channels
    if num_ortho is None: num_ortho = n
    if(bias):
        m = m + 1
    num_free_weights = n*m-((n*n+n)/2)
    free_weights = tf.get_variable(name='free', shape=[num_free_weights + n],
                                   initializer=tf.random_uniform_initializer(-1, 1))
    id_mat = tf.eye(m, dtype=tf.float32)
    initial_mat = tf.eye(m, n)
    vec_sizes = tf.range(m-num_ortho+1, m+1)
    start_indices = tf.cumsum(vec_sizes, exclusive=True)

    def find_HHvecs(start_idx, vec_size):
        free_weights_slice = tf.slice(free_weights, [start_idx], [vec_size])
        paddings = [[m-vec_size, 0]]

        weights_vec = tf.pad(free_weights_slice, paddings, "CONSTANT")
        weights_vec = tf.nn.l2_normalize(weights_vec, dim=0)
        return weights_vec

    weights_vecs = tf.map_fn(lambda x: find_HHvecs(x[0], x[1]), (start_indices, vec_sizes), dtype=tf.float32)
    ortho_matrices = tf.expand_dims(id_mat, axis=0) - 2*(tf.expand_dims(weights_vecs, axis=2)*tf.expand_dims(weights_vecs, axis=1))

    weights_matrix = tf.foldl(tf.matmul, ortho_matrices, parallel_iterations=min(n/2, 32))
    filter_tensor = tf.matmul(weights_matrix, initial_mat)
    bias_tensor = tf.zeros([1, n])
    if(bias):
        filter_tensor, bias_tensor = tf.slice(filter_tensor, [0, 0], [m-1, n]), tf.slice(filter_tensor, [m-1, 0], [1, n])
    filter_tensor = tf.reshape(filter_tensor, shape=[in_height, in_width, in_channels, out_channels]) 
    conv_filter = tf.get_variable(name='conv_filter', 
                                  shape=[in_height, in_width, in_channels, out_channels], 
                                  trainable=False, initializer=tf.zeros_initializer)
    conv_bias = tf.get_variable(name='conv_bias', 
                                shape=[1, out_channels], trainable=False, 
                                initializer=tf.zeros_initializer)

    conv_filter = conv_filter.assign(filter_tensor)
    conv_bias = conv_bias.assign(bias_tensor)
    tf.add_to_collection('ortho_compute', filter_tensor)
    tf.add_to_collection('ortho_compute', bias_tensor)
    tf.add_to_collection('ortho_assign', conv_filter)
    tf.add_to_collection('ortho_assign', conv_bias)
    return filter_tensor, bias_tensor


def main():
    # my code here
    a = orthoconv_filter(3, 3, 16, 32, bias=True)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        x,y, = sess.run([a[0], a[1]])
        check(x,y)

if __name__ == "__main__":
    main()
