from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

def conv2d(x, input_filters, output_filters, strides, kernel):
	with tf.variable_scope('conv'):
		w_shape = [kernel, kernel, input_filters, output_filters]
		weight = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.1), name = 'weight')

		#x_padded = tf.pad(x, tf.to_int64([[0, 0], [kernel / 2, kernel / 2], [kernel / 2, kernel / 2], [0, 0]]), mode = 'REFLECT')

		return tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME', name='conv')

def norm(x):
	epsilon = 1e-9
	mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

	return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def residual(x, filters, kernel, strides):
	with tf.variable_scope('residual'):
		conv1 = tf.nn.relu(conv2d(x, filters, filters, strides, kernel))
		conv2 = conv2d(conv1, filters, filters, strides, kernel)

	residual = x + conv2
	return tf.nn.relu(residual)

def deconv2d(x, input_filters, output_filters, strides, kernel, output_shape):
	with tf.variable_scope('deconv'):
		w_shape = [kernel, kernel, output_filters, input_filters]
		weight = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.1), name='weight')

		return tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], padding='SAME', name='deconv')


def net(x):
	
	output_shape_1 = x.get_shape()
	with tf.variable_scope('conv1'):
		conv1 = tf.nn.relu(norm(conv2d(x, 1, 32, 1, 9)))

	output_shape_2 = conv1.get_shape()        
	with tf.variable_scope('conv2'):
		conv2 = tf.nn.relu(norm(conv2d(conv1, 32, 64, 2, 3)))
	
	output_shape_3 = conv2.get_shape()        
	with tf.variable_scope('conv3'):
		conv3 = tf.nn.relu(norm(conv2d(conv2, 64, 128, 2, 3)))
        
	with tf.variable_scope('res1'):
		res1 = residual(conv3, 128, 3, 1)
        
	with tf.variable_scope('res2'):
		res2 = residual(res1, 128, 3, 1)
        
	with tf.variable_scope('res3'):
		res3 = residual(res2, 128, 3, 1)

	with tf.variable_scope('res4'):
		res4 = residual(res3, 128, 3, 1)

	with tf.variable_scope('res5'):
		res5 = residual(res4, 128, 3, 1)
        
	with tf.variable_scope('deconv1'):
		deconv1 = tf.nn.relu(norm(deconv2d(res5, 128, 64, 2, 3, output_shape_3)))
        
	with tf.variable_scope('deconv2'):
		deconv2 = tf.nn.relu(norm(deconv2d(deconv1, 64, 32, 2, 3, output_shape_2)))
        
	with tf.variable_scope('deconv2'):
		deconv3 = tf.nn.relu(norm(deconv2d(deconv2, 32, 1, 1, 9, output_shape_1)))
        
	return deconv3
