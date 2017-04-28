import numpy as np
import tensorflow as tf
import os
import csv
from multi_bp import *


x_raw = inputx('2.csv')
y_raw = inputy('1.csv')
x_data = np.array(x_raw).astype(np.float32)
y_data = np.array(y_raw).astype(np.float32)
scalery = preprocessing.StandardScaler().fit(y_data)
scaler = preprocessing.StandardScaler().fit(x_data)
#x_data_standard = scaler.transform(x_data)
x_data = scaler.transform(x_data)

xs = tf.placeholder(tf.float32,[None,12])
ys = tf.placeholder(tf.float32,[None,1])

l1 = addLayer(xs,12,24,activity_function=tf.nn.relu)
l2 = addLayer(l1,24,12, activity_function=tf.nn.relu)
l3 = addLayer(l2,12,1,activity_function=None)

predict = l3 * scalery.scale_ + scalery.mean_

saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "tmp/model1.ckpt")
    print "Model restored."
    lable_y =  sess.run(predict, feed_dict={xs: x_data, ys: y_data})
testanalysis(np.array(y_raw).astype(np.float32), lable_y, 0.05)