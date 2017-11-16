import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import collections, csv, os
from scipy.ndimage.filters import gaussian_filter
import random
n_steps = 24 * 7
n_inputs = 1
n_neurons = 100
n_outputs = 1
learning_rate = 0.0001
n_iterations = 2000
batch_size = 50
n_layers = 3

ARTIFACTS = "../calcs/la-fire-rnn-smoke"
DATASET = "../ds/los-angeles-fire-counts-2001-2017-hours.csv"

with open(DATASET, "rt") as f:
    reader = csv.DictReader(f)
    data = []
    for i, line in enumerate(reader):
        data.append(float(line["count"]))
    data = np.array(data)
    tsset = gaussian_filter(data, sigma=2)

t_instance = np.linspace(0, n_steps + 1, n_steps + 1)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def next_batch(n_batch, n_steps):
    count = len(data)
    ys = []
    for b in range(0, batch_size):
        i = random.randint(0, count - n_steps * 2)
        run = []
        for s in range(0, n_steps * 2 + 1):
            run.append(data[i])
            i += 1
        ys.append(run)
    ys = np.array(ys)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

if not os.path.isfile(ARTIFACTS + ".index"):
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(batch_size, n_steps)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tLoss:", mse)

        saver.save(sess, ARTIFACTS)

bX, by = next_batch(1, n_steps)

with tf.Session() as sess:
    saver.restore(sess, ARTIFACTS)
    X_new = bX
    y_pred = sess.run(outputs, feed_dict={X: X_new})

plt.title("LA Fire Department Incidents", fontsize=14)
plt.plot(t_instance[1:], by[0,:,0], "g-", markersize=2, label="Target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r-", markersize=1, label="Prediction")
plt.legend(loc="upper left")
plt.xlabel("Day")

plt.show()