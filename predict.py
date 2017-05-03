import csv
import numpy as np
import tensorflow as tf
from sklearn import linear_model
from sklearn import preprocessing
import os
from multi_bp import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

x_raw = inputx('2.csv')
y_raw = inputy('1.csv')
x_raw = np.array(x_raw).astype(np.float32)
y_raw = np.array(y_raw).astype(np.float32)
#scaler = preprocessing.StandardScaler().fit(x_data)
#print scaler.mean_, scaler.scale_
#x_data = scaler.transform(x_data)
#scalery = preprocessing.StandardScaler().fit(y_data)
#print scalery.mean_, scalery.scale_
#y_data = scalery.transform(y_data)
#y_raw = np.array(y_raw).astype(np.float32)

xs = tf.placeholder(tf.float32,[None,12])
ys = tf.placeholder(tf.float32,[None,1])

#predict = l3 * scalery.scale_ + scalery.mean_


# Restore variables from disk.
print "Model restored."
x = np.linspace(0,0.2,200)
y = []
for i in range(len(x)): y.append(0);
plt.figure(figsize=(10, 5))
plt.xlabel('theta')
plt.ylabel('ratio')

xs = tf.placeholder(tf.float32,[None,12])
ys = tf.placeholder(tf.float32,[None,1])

l1 = addLayer(xs,12,36,activity_function=tf.nn.relu)
#l2 = addLayer(l1,20,20, activity_function=tf.nn.relu)
l3 = addLayer(l1, 36, 1, activity_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys-l3)),reduction_indices = [1]))

saver = tf.train.Saver()
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

train = tf.train.AdamOptimizer(0.001).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.4, random_state=0)
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(x_raw, y_raw):
    x_train, x_test = x_raw[train_index], x_raw[test_index]
    y_train, y_test = y_raw[train_index], y_raw[test_index]
    y_raw_train = y_train
    y_raw_test = y_test
    scaler = preprocessing.StandardScaler().fit(x_test)
    x_test = scaler.transform(x_test)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    scaler = preprocessing.StandardScaler().fit(y_train)
    y_train = scaler.transform(y_train)
    scalery = preprocessing.StandardScaler().fit(y_test)
    y_test = scalery.transform(y_test)
    predict = l3 * scalery.scale_ + scalery.mean_
        #print y_raw_test
    for i in range(50000):
            # train = tf.train.GradientDescentOptimizer(1.0 / (1.0 * (i + 10))).minimize(loss)
        sess.run(train, feed_dict={xs: x_train, ys: y_train})
            # print sess.run(l2, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print sess.run(loss, feed_dict={xs: x_train, ys: y_train})
            save_path = saver.save(sess, 'tmp/model1.ckpt')

    predict = l3 * scalery.scale_ + scalery.mean_
    lable_y = sess.run(predict, feed_dict={xs: x_test, ys: y_test})
    for i in range(len(x)): y[i] += testanalysis(y_raw_test, lable_y, x[i]) * len(lable_y) / 4729;
    #print testanalysis(y_raw_test, lable_y, 0.1) * len(lable_y) / 4729;

print len(y),y
'''
    for i in range(len(x)):
        y.append(testanalysis(np.array(y_raw).astype(np.float32), lable_y, x[i]))
    '''

plt.plot(x, y)
plt.show()
print y