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




if __name__ == "__main__":
    print ('This is main of module "hello.py"')
    x_raw = inputx('2.csv')
    y_raw = inputy('1.csv')
    x_data = np.array(x_raw).astype(np.float32)
    #x_data = np.array(prepare(x_raw)).astype(np.float32)
    y_data = np.array(y_raw).astype(np.float32)
    scaler = preprocessing.StandardScaler().fit(x_data)
    print scaler.mean_, scaler.scale_
    #x_data_standard = scaler.transform(x_data)
    x_data = scaler.transform(x_data)
    scalery = preprocessing.StandardScaler().fit(y_data)
    print scalery.mean_, scalery.scale_
    # x_data_standard = scaler.transform(x_data)
    y_data = scalery.transform(y_data)



    xs = tf.placeholder(tf.float32,[None,12])
    ys = tf.placeholder(tf.float32,[None,1])

    l1 = addLayer(xs,12,24,activity_function=tf.nn.relu)
    l2 = addLayer(l1,24,1,activity_function=None)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys-l2)),reduction_indices = [1]))


    saver = tf.train.Saver()
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')

        #train =  tf.train.GradientDescentOptimizer(0.003).minimize(loss)
    train = tf.train.AdamOptimizer(0.005).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(10000):
        sess.run(train,feed_dict={xs:x_data,ys:y_data})
        if i%50 == 0:
            print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
            save_path = saver.save(sess, 'tmp/model.ckpt')

    #self.session.run(self.output_layer, feed_dict={self.input_layer: case})
    print("path: %s"%(save_path))
    sess.run()

