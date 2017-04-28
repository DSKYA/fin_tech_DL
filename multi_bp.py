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




def make_layer(inputs, in_size, out_size, activate=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    basis = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    result = tf.matmul(inputs, weights) + basis
    if activate is None:
        return result
    else:
        return activate(result)


class BPNeuralNetwork:
    def __init__(self):
        self.session = tf.Session()
        self.loss = None
        self.optimizer = None
        self.input_n = 0
        self.hidden_n = 0
        self.hidden_size = []
        self.output_n = 0
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.label_layer = None

    def __del__(self):
        self.session.close()

    def setup(self, ni, nh, no):
        # set size args
        self.input_n = ni
        self.hidden_n = len(nh)  # count of hidden layers
        self.hidden_size = nh  # count of cells in each hidden layer
        self.output_n = no
        # build input layer
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_n])
        # build label layer
        self.label_layer = tf.placeholder(tf.float32, [None, self.output_n])
        # build hidden layers
        in_size = self.input_n
        out_size = self.hidden_size[0]
        inputs = self.input_layer
        self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu))
        for i in range(self.hidden_n-1):
            in_size = out_size
            out_size = self.hidden_size[i+1]
            inputs = self.hidden_layers[-1]
            self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu))
        # build output layer
        self.output_layer = make_layer(self.hidden_layers[-1], self.hidden_size[-1], self.output_n)

    def train(self, cases, labels, limit=10000, learn_rate=0.01):
        train = tf.train.AdamOptimizer(0.005).minimize(loss)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        for i in range(10000):
            sess.run(train, feed_dict={self.input_layer: cases, self.label_layer: labels})
            if i % 50 == 0:
                print sess.run(loss, feed_dict={self.input_layer: cases, self.label_layer: labels})
        '''        
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.label_layer - self.output_layer)), reduction_indices=[1]))
        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        initer = tf.initialize_all_variables()
        # do training
        self.session.run(initer)
        for i in range(limit):
            self.session.run(self.optimizer, feed_dict={self.input_layer: cases, self.label_layer: labels})
            if i % 50 == 0:
                print self.session.run(self.loss, feed_dict={self.input_layer: cases, self.label_layer: labels})
        '''

    def predict(self, case):
        return self.session.run(self.output_layer, feed_dict={self.input_layer: case})

    def test(self):
        print ('This is main of module "hello.py"')
        x_raw = inputx('2.csv')
        y_raw = inputy('1.csv')
        x_data = np.array(x_raw).astype(np.float32)
        # x_data = np.array(prepare(x_raw)).astype(np.float32)
        y_data = np.array(y_raw).astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(x_data)
        print scaler.mean_, scaler.scale_
        # x_data_standard = scaler.transform(x_data)
        x_data = scaler.transform(x_data)
        #scalery = preprocessing.StandardScaler().fit(y_data)
        #print scalery.mean_, scalery.scale_
        #y_data = scalery.transform(y_data)

        #x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        #y_data = np.array([[0, 1, 1, 0]]).transpose()
        test_data = np.array([x_raw[0]])
        self.setup(12, [20, 1], 1)
        self.train(x_data, y_data)
        #print (self.predict(test_data) * scalery.scale_ + scalery.mean_)
        print (self.predict(test_data))

nn = BPNeuralNetwork()
nn.test()
