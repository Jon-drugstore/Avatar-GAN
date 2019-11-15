import tensorflow as tf
import tensorflow.contrib.slim as slim


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5, scale=True,
                                        scope=name)


def instance_norm(x, name="instance_norm"):
    with tf.variable_scope(name):
        depth = x.get_shape()[3]
        scale = tf.get_variable(name="scale",
                                shape=[depth],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))

        offset = tf.get_variable(name="offset",
                                 shape=[depth],
                                 initializer=tf.constant_initializer(0.0))

        mean, variance = tf.nn.moments(x,
                                       axes=[1, 2],
                                       keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv

        return scale * normalized + offset


def conv2d(x, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(x, output_dim, ks, s,
                           padding=padding,
                           activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def deconv2d(x, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(x, output_dim, ks, s,
                                     padding='SAME',
                                     activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def linear(x, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(name="Matrix",
                                 shape=[x.get_shape()[-1], output_size],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable(name="bias",
                               shape=[output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias
