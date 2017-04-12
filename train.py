from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import model_wei
import data_process
import reader
import time
import os

Batch_size = 2
Epoch = 1000
band = '1'
patch = '5'

def main():
	for d in ['/gpu:0', '/gpu:1']:
		with tf.device(d):
			#real all signal
			all_signal = tf.placeholder(tf.float32, shape=[Batch_size, 1024, 1024 ,1])

			#get the immitated noise signal by real all signal thorough net
			with tf.variable_scope('get_noise'):
				noise_signal_immitate = model_wei.net(all_signal)

			#the real noise signal
			noise_signal_real = tf.placeholder(tf.float32, shape=[Batch_size, 1024, 1024, 1])

			#loss of noise part. Compute the square error of immitation_noise and real_noise
			loss_noise_part = (tf.nn.l2_loss(noise_signal_real - noise_signal_immitate) * 2) / tf.to_float(tf.size(noise_signal_real))

			#Get the cmd_immitation by noise_immitation thorough net
			with tf.variable_scope('get_cmd'):
				cmd_signal_immitate = model_wei.net(all_signal - noise_signal_immitate)

			#the real cmd signal
			cmd_signal_real = tf.placeholder(tf.float32, shape=[Batch_size, 1024, 1024, 1])

			#loss for cmd part. Compute the square error of immitation_cmd and real_cmd
			loss_cmd_part = (tf.nn.l2_loss(cmd_signal_real - cmd_signal_immitate) * 2) / tf.to_float(tf.size(cmd_signal_real))
					
			#Sum tow loss as one
			loss_all = loss_noise_part + loss_cmd_part

			train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_all)

	saver = tf.train.Saver()
	#initialization
	init = tf.global_variables_initializer()

	#all_data contains cmd, all and noise file pathes 
	all_data = data_process.do_it(reader.load_data(band, patch))

	acount = all_data._numbers

	with tf.Session() as sess:
		sess.run(init)
		total_batch = int(acount / Batch_size)
		iteration = 0
		print ("Train Begins!")
		start_time = time.time()
		for i in range(Epoch):
			for j in range(total_batch):
				#get a batch of data to feed
				batch_all, batch_noise, batch_cmd = all_data.next_batch(Batch_size)

				feed_dict = {all_signal: batch_all, noise_signal_real: batch_noise, cmd_signal_real: batch_cmd}
					
				#One iteration
				iteration += 1
				_, loss_iter = sess.run([train_op, loss_all], feed_dict=feed_dict)

				if iteration % 20 == 0:
					timer = time.strftime("%Y-%m-%d %H:%M:%S")
					print('Time is %s.'%(timer), 'At iteration :%d' % (iteration), 'loss at this time: {:.9f}'.format(loss_iter))
				if iteration % 200 == 0:
					print('\nEvery iter takes: {:.9f}\n'.format((time.time() - start_time) / float(200)))
					start_time = time.time()
				if iteration % 10000 == 0:
					saver.save(sess, 'model/denoise-model.ckpt', global_step=iteration)
		saver.save(sess, 'model/denoise-model-done.ckpt')
		print ("Train ends! And model has been save!")

if __name__ == '__main__':
	main()
