import csv
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def inputx(s):
    """
    getsample from 2.csv
    :param s: path
    :return: sample list
    """
    n_row = 0
    x_data = []
    with open(s, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if n_row != 0:
                x_data.append(row[1:3] + row[4:5] + row[6:len(row) - 1])
            n_row += 1
    return x_data


def inputy(s):
    """
    getlabels from 1.csv
    :param s: path
    :return: labels list
    """
    n_row = 0
    x_data = []
    with open(s, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if n_row != 0:
                x_data.append(row[1:len(row) - 1])
            n_row += 1
    return x_data

    """
        add cell in ANN
        :param inputData: last layer's output
        :param inSize:  number of last layer's cells 
        :param outSize: number of this layer's cells
        :param activity_function: whether use activity_function
        :return: 
    """
def addLayer(inputData,inSize,outSize,activity_function = None):
	Weights = tf.Variable(tf.random_normal([inSize,outSize]))
	basis = tf.Variable(tf.zeros([1,outSize])+0.1)
	weights_plus_b = tf.matmul(inputData,Weights)+basis
	if activity_function is None:
		ans = weights_plus_b
	else:
		ans = activity_function(weights_plus_b)
	return ans

def getcrossvalidator(train_index, test_index,x_raw, y_raw):
    """
    10-fold cross-validator
    :param train_index: index of each traindata
    :param test_index: index of each testdata
    :param x_raw:   samples' raw data
    :param y_raw:   labels' raw data
    :return: x_train, x_test,y_train, y_test, y_raw_train, y_raw_test, scalery.scale_, scalery.mean_
    """
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
    return x_train, x_test,y_train, y_test, y_raw_train, y_raw_test, scalery.scale_, scalery.mean_

def testanalysis(y_data,test_y,theta):
    """
    count accurate datas' ratio
    :param y_data: origin labels
    :param test_y: predict labels
    :param theta: theta
    :return: acc ratio
    """
    num = 0
    account = 0
    for i in range(len(y_data)):
        if (abs(test_y[i, 0] - y_data[i,0]) / y_data[i,0] <= theta):
            account += 1
        num += 1
    return account * 1.0 / num

x_raw = inputx('2.csv')
y_raw = inputy('1.csv')
x_raw = np.array(x_raw).astype(np.float32)  # change to array
y_raw = np.array(y_raw).astype(np.float32)

xs = tf.placeholder(tf.float32, [None, 12])  # input layer
ys = tf.placeholder(tf.float32, [None, 1])  # output layer

x = np.linspace(0, 0.5, 500)  # theta from 0 to 0.5 each is 0.001
y = []
for i in range(len(x)): y.append(0);  # init ratio[]

l1 = addLayer(xs, 12, 24, activity_function=tf.nn.relu)     #add three layers
#l2 = addLayer(l1,60,20, activity_function=tf.nn.relu)
l3 = addLayer(l1, 24, 1, activity_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - l3)), reduction_indices=[1]))       #define loss function

saver = tf.train.Saver()        #makedir model's save path
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

train = tf.train.AdamOptimizer(0.002).minimize(loss)    #use AdamOptimizer,rateis 0.001
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(x_raw, y_raw):
    x_train, x_test,y_train, y_test,y_raw_train,y_raw_test,y_scale,y_mean = getcrossvalidator(train_index,test_index,x_raw,y_raw)
    for i in range(5000):
        sess.run(train, feed_dict={xs: x_train, ys: y_train})
        if i % 50 == 0:
            print sess.run(loss, feed_dict={xs: x_train, ys: y_train})
            #save_path = saver.save(sess, 'tmp/model2.ckpt')
    save_path = saver.save(sess, 'tmp/model.ckpt')
    predict = l3 * y_scale + y_mean     #predict output
    lable_y = sess.run(predict, feed_dict={xs: x_test, ys: y_test})     #get all testdatas' predict
    for i in range(len(x)): y[i] += testanalysis(y_raw_test, lable_y, x[i]) * len(lable_y) / len(x_raw);

plt.figure(figsize=(10, 5))     #paint
plt.xlabel('theta')
plt.ylabel('ratio')
plt.plot(x, y)
plt.show()
