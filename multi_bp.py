import csv
import numpy as np
import tensorflow as tf
from sklearn import linear_model
from sklearn import preprocessing
import os





def inputx(s):
    n_row = 0
    x_data = []
    with open(s, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if n_row != 0:
                #x_data.append(row[1:len(row) - 1])
                x_data.append(row[1:3] + row[4:5] + row[6:len(row) - 1])
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
    for i in range(len(x)):
        for j in range(len(x[0])):
            if(float(x[i][j]) == 0):
                x[i][j] = tsum[j]
    return x







def addLayer(inputData,inSize,outSize,activity_function = None):
	Weights = tf.Variable(tf.random_normal([inSize,outSize]))
	basis = tf.Variable(tf.zeros([1,outSize])+0.1)
	weights_plus_b = tf.matmul(inputData,Weights)+basis
	if activity_function is None:
		ans = weights_plus_b
	else:
		ans = activity_function(weights_plus_b)
	return ans


def testanalysis(y_data,test_y,theta):
    num = 0
    account = 0
    for i in range(len(y_data)):
        if (abs(test_y[i, 0] - y_data[i,0]) / y_data[i,0] <= theta):
            account += 1
        num += 1
    #print "ac ratio is : %s" % (account * 1.0 / num)
    return account * 1.0 / num



if __name__ == "__main__":
    x_raw = inputx('2.csv')
    y_raw = inputy('1.csv')
    x_data = np.array(x_raw).astype(np.float32)
    y_data = np.array(y_raw).astype(np.float32)
    scaler = preprocessing.StandardScaler().fit(x_data)
    print scaler.mean_, scaler.scale_
    x_data = scaler.transform(x_data)
    scalery = preprocessing.StandardScaler().fit(y_data)
    print scalery.mean_, scalery.scale_
    y_data = scalery.transform(y_data)

    xs = tf.placeholder(tf.float32,[None,12])
    ys = tf.placeholder(tf.float32,[None,1])

    l1 = addLayer(xs,12,20,activity_function=tf.nn.relu)
    l2 = addLayer(l1,20,20, activity_function=tf.nn.relu)
    l3 = addLayer(l2, 20, 1, activity_function=None)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys-l3)),reduction_indices = [1]))
    predict = l3 * scalery.scale_ + scalery.mean_

    saver = tf.train.Saver()
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')
    '''
    else:
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, "/tmp/model.ckpt")
            print "Model restored."
    '''
    #train =  tf.train.GradientDescentOptimizer(0.002).minimize(loss)
    train = tf.train.AdamOptimizer(0.001).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(50000):
        #train = tf.train.GradientDescentOptimizer(1.0 / (1.0 * (i + 10))).minimize(loss)
        sess.run(train,feed_dict={xs:x_data,ys:y_data})
        #print sess.run(l2, feed_dict={xs: x_data, ys: y_data})
        if i%50 == 0:
            print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
            save_path = saver.save(sess, 'tmp/model1.ckpt')

    #print sess.run(predict, feed_dict={xs:np.array(test_x[0])})
    lable_y =  sess.run(predict, feed_dict={xs: x_data, ys: y_data})
    print lable_y
    print("path: %s"%(save_path))
    testanalysis(np.array(y_raw).astype(np.float32), lable_y, 0.01)

