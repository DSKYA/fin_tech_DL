#!/usr/bin/env python
# -*- coding:utf-8 -*-
import csv
import numpy as np
import tensorflow as tf
from sklearn import linear_model
from sklearn import preprocessing

def inputx(s):
    n_row = 0
    x_data = []
    with open(s, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if n_row != 0:
                x_data.append(row[1:len(row) - 1])
                #x_data.append(row[1:3] + row[4:5] + row[6:len(row) - 1])
                #print x_data
            #print row
            n_row += 1
    #x = x_data
    #print x_data
    return x_data

def inputy(s):
    n_row = 0
    x_data = []
    with open(s, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if n_row != 0:
                x_data.append(row[1:len(row) - 1])
                #x_data.append(row[1:3] + row[4:5] + row[6:len(row) - 1])
                #print x_data
            #print row
            n_row += 1
    #x = x_data
    #print x_data
    return x_data


def prepare(x):
    tsum = [0] * len(x[0])
    count = [0] * len(x[0])
    a = np.array(x).T.astype(np.float32)
    y = a.tolist()
    for i in range(len(count)):
        count[i] = len(y[i]) - y[i].count(0)
        tsum[i] = sum(y[i]) / count[i]
        print ("%s = %s" %(i,1.0 * y[i].count(0) / 4729))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if(float(x[i][j]) == 0):
                x[i][j] = tsum[j]
    return x

def testanalysis(theta,x_data,y_data,w_x,w_b):
    #w_x = np.mat(w_x)
    x_data = np.mat(x_data.T.tolist())
    test_y =  w_x.T * x_data
    num = 0
    account = 0
    y_ori = y_data.tolist()
    for i in range(len(y_ori)):
        if(abs(test_y[0,i] + w_b[0]  - y_data[i]) / y_data[i] <= theta):
            account += 1
        num += 1
    print "ac ratio is : %s" % (account * 1.0 / num)

if __name__ == "__main__":
    print ('This is main of module "hello.py"')
    x_raw = inputx('2.csv')
    y_raw = inputy('1.csv')
    #x_data = np.array(x_raw).astype(np.float32)
    x_data = np.array(prepare(x_raw)).astype(np.float32)
    y_data = np.array(y_raw).astype(np.float32)
    scaler = preprocessing.StandardScaler().fit(x_data)
    print scaler.mean_, scaler.scale_
    x_data_standard = scaler.transform(x_data)

    #W = tf.Variable(tf.zeros([12, 1]))
    W = tf.Variable(tf.zeros([14, 1]))
    b = tf.Variable(tf.zeros([1, 1]))
    y = tf.matmul(x_data_standard, W) + b

    loss = tf.reduce_mean(tf.square(y - y_data.reshape(-1, 1))) / 2
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for step in range(1000):
        sess.run(train)
        if step % 10 == 0:
            print step, sess.run(W).flatten(), sess.run(b).flatten()
    print "Coefficients of tensorflow (input should be standardized): K=%s, b=%s" % (
    sess.run(W).flatten(), sess.run(b).flatten())
    print "Coefficients of tensorflow (raw input): K=%s, b=%s" % (
    sess.run(W).flatten() / scaler.scale_, sess.run(b).flatten() - np.dot(
    scaler.mean_ / scaler.scale_, sess.run(W)))

    reg = linear_model.LinearRegression()
    reg.fit(x_data, y_data)
    print "Coefficients of sklearn: K=%s, b=%f" % (reg.coef_, reg.intercept_)

    testanalysis(0.01,x_data,y_data,sess.run(W).flatten() / scaler.scale_, sess.run(b).flatten() - np.dot(scaler.mean_ / scaler.scale_, sess.run(W)))
    testanalysis(0.01,x_data,y_data,reg.coef_.T, reg.intercept_)