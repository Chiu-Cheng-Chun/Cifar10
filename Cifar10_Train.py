
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.font_manager as fm

#讓matplotlib補上中文顯示
myfont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig = plt.figure()

#結構初始化
tf.reset_default_graph()

t0 = time.clock()

"""
定義訓練集、測試集
"""

X_Train = np.load( "Cifar10_X_Train.npy" )
Y_Train = np.load( "Cifar10_Y_Train.npy" )

X_Test = np.load( "Cifar10_X_Test.npy" )
Y_Test = np.load( "Cifar10_Y_Test.npy" )

Label_Mean = np.load( "Cifar10_Label_Mean.npy" )

X_Train_with_Sobel = np.load( "Cifar10_X_Train_with_Sobel.npy" )
X_Test_with_Sobel = np.load( "Cifar10_X_Test_with_Sobel.npy" )

"""
定義變數
"""
W = []
b = []

"""
batch normalization
"""
scale = []
shift = []

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

"""
定義卷積
"""
def Convolution(inputs, kernel_size):
    kernel = tf.Variable(tf.truncated_normal(kernel_size, stddev=0.5))
    return tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')

"""
定義池化
"""
def Max_pool(conv):
    return tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')

"標準化"
def Norm(pool):
    return tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')


epoches = 4500  #設定迭代次數
learning_rate = 0.01 #學習速率

"""
卷積格式:Convolution(輸入矩陣, filter個數, filter大小)
池化格式:Max_pool(輸入矩陣)
"""
num_filters = [64, 64]
h1 = Convolution(x, [5, 5, 3, num_filters[0]])#第一次卷積
p1 = Max_pool(h1)#第一次池化
n1 = Norm(p1)
h2 = Convolution(n1, [5, 5, num_filters[0], num_filters[1]])#第二次卷積
p2 = Max_pool(h2)#第二次池化
n2 = Norm(p2)

"""
將池化後的3維矩陣攤平成2維矩陣
"""
pool_times = 2
pool_flat = tf.reshape(n2, [-1, 8 * 8 * num_filters[-1]])
input_size = 8 * 8 * num_filters[-1]

n_neurons = [input_size, 512, 256, 10]

"""
建構權重及偏差
"""
for i in range(0, len(n_neurons)-1):
    W.append(tf.Variable(tf.truncated_normal([n_neurons[i],n_neurons[i+1]])))
    b.append(tf.Variable(tf.zeros([1.0 ,n_neurons[i+1]])))
    scale.append(tf.Variable(tf.ones([n_neurons[i+1]])))
    shift.append(tf.Variable(tf.zeros([n_neurons[i+1]])))
    
"""
定義Batch Normalization
"""
def Batch_norm(Wx_plus_b, i):
    fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0, 1])
    Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift[i], scale[i], 10**(-3))
    return Wx_plus_b

"""
定義全聯接層模型
"""
def fc_model(X, n_neurons, W, b):
    X = tf.layers.batch_normalization(X, training=True)
    for i in range(0, len(n_neurons) - 2):
        X = tf.nn.relu(tf.matmul(X,W[i]) + b[i])
        X = Batch_norm(X, i)
    result = tf.matmul(X, W[-1]) + b[-1]
    return result

"""
定義損失函數，此為交叉熵代價函數
"""
def Cost(y_label, prediction):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_label))
    return cross_entropy

prediction = fc_model(pool_flat, n_neurons, W, b)
cross_entropy = Cost(y, prediction)
train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#--------------------------------------------train-----------------------------------------

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

costs = np.ones((epoches+1,1))
batch_size = 500
"""
開始訓練
"""
for i in range(epoches+1):
    r = int(i % ((len(X_Train) - batch_size) / batch_size))
    batch_xs, batch_ys = X_Train[r*batch_size:(r+1)*batch_size], Y_Train[r*batch_size:(r+1)*batch_size]
    sess.run(train, feed_dict = {x: batch_xs, y: batch_ys})
    costs[i] = sess.run(cross_entropy, feed_dict = {x: batch_xs, y: batch_ys})
    if i % 100 == 0:
        print(i,"epoch(es)", " Done!", "Cost:", costs[i])

#plot
x_axis = np.arange(epoches)
fig, ax1 = plt.subplots()  
ax1.plot(x_axis, costs[1:], "green", label = "Cost")
ax1.set_ylabel('Cost值',fontproperties=myfont)
ax1.set_xlabel('訓練次數',fontproperties=myfont)
ax1.legend(loc=1,prop=myfont)
plt.show()

"""
計算訓練集準確度
"""
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_batch = 50
train_acc = []
ran = len(X_Train) / train_batch
for i in range(train_batch):
    train_acc = np.append(train_acc, accuracy.eval(feed_dict={x: X_Train[int(i*ran):int((i+1)*ran)], y: Y_Train[int(i*ran):int((i+1)*ran)]}))

print("Train Set Accuracy:", np.mean(train_acc) * 100, "%")

"""
計算測試集準確度
"""
test_batch = 10
test_acc = []
ran = len(X_Test) / test_batch
for i in range(test_batch):
    test_acc = np.append(test_acc, accuracy.eval(feed_dict={x: X_Test[int(i*ran):int((i+1)*ran)], y: Y_Test[int(i*ran):int((i+1)*ran)]}))

print("Test Set Accuracy:", np.mean(test_acc) * 100, "%")
    
print("總共費時:",time.clock()-t0,"秒")
