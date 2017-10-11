from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([5, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.array([[0.2707,0.4433,0.1505,0.3709],
[0.4727,0.5118,0.5516,0.5063],
[0.6529,0.5959,0.5622,0.526],
[0.5392,0.4217,0.4048,0.4148],
[0.4681,0.5276,0.4826,0.4278]])

y_data = np.array([[0.2862,0.3178,0.4435,0.4866,0.2784,0.1882,0.4408,0.4311,0.1427,0.0588,0.381,0.4884,0.1169,0.1846,0.3283,0.3574],
[0.4746,0.3958,0.4323,0.5243,0.4458,0.5894,0.622,0.511,0.4715,0.521,0.5473,0.4101,0.579,0.5542,0.5394,0.4838],
[0.6385,0.4538,0.4087,0.5574,0.6176,0.7348,0.6953,0.5228,0.6962,0.7563,0.7661,0.6107,0.4209,0.3497,0.3139,0.3821],
[0.4434,0.562,0.4143,0.3962,0.6091,0.6303,0.4124,0.422,0.3959,0.4445,0.3481,0.4338,0.3351,0.3889,0.407,0.4417],
[0.4676,0.423,0.4974,0.5771,0.4263,0.554,0.4461,0.5491,0.3642,0.5208,0.555,0.3859,0.5765,0.4219,0.3669,0.3717]]
)

# print(x_data)
# print(y_data)

# placeholder
xs = tf.placeholder(tf.float32, [None, 4])
ys = tf.placeholder(tf.float32, [None, 16])

# hidden layer
l1 = add_layer(xs, 4, 4, activation_function=tf.nn.relu)
# output layer
prediction = add_layer(l1, 4, 16, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        print(prediction_value)
