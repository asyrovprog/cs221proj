import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv, os, random

# do we need to rebuild artifacts (weights)?:
rebuild_artifacts = True
# do we need to build logs for tensorboard?:
build_tensorboard_logs = False

# number of previous values, since we look at 1 week here to predict next week
# our window is 24 by 7. we will pass all previous days as state.
state_vector_size = 24 * 7
# we have just one input, which is number of emergency phone calls during current hour:
input_vector_size = 1
# number of steps and size of prediction vector:
neuron_count = state_vector_size + input_vector_size
# our prediction is again number of phone calls for hour:
output_vector_size = 1
# learning rate found during optimization in dev phase:
learning_rate = 0.0005
# iteration count found during optimization in dev phase:
n_iterations = 800
# iteration count found during optimization in dev phase:
batch_size = 50
# number of layers found during optimization in dev phase:
layer_count = 2

# prefix for file names for artifacts (weights, etc), if we have them we can skip training phase
# and just load them and to prediction
ARTIFACTS = "../calcs/sf-fire-rnn-smoke"

# name of training dataset
DATASET = "../ds/sf-fire-counts-train.csv"

# name of test dataset (it should be *dev.* except for final phase)
TARGET_DATASET = "../ds/sf-fire-counts-dev.csv"

data = []
t_instance = np.linspace(0, state_vector_size + 1, state_vector_size + 1)

# load data as array of counts of incidents per hour
def load_dataset(filename):
    global data
    with open(filename, "rt") as f:
        reader = csv.DictReader(f)
        data = []
        for i, line in enumerate(reader):
            data.append(float(line["count"]))
        data = np.array(data)

# initialize tensorflow default graph
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# this returns 'step_count' batches, each of 'batch_size'
def next_batch(batch_size, step_count):
    count = len(data)
    ts = []
    for b in range(0, batch_size):
        i = random.randint(0, count - step_count * 2 - 1)
        run = []
        for s in range(0, step_count * 2 + 1):
            run.append(data[i])
            i += 1
        ts.append(run)
    ts = np.array(ts)
    return ts[:, :-1].reshape(-1, step_count, 1), ts[:, 1:].reshape(-1, step_count, 1)

# main function which trains and predicts
def train_predict():
    # load training dataset
    load_dataset(DATASET)

    X = tf.placeholder(tf.float32, [None, state_vector_size, input_vector_size])
    y = tf.placeholder(tf.float32, [None, state_vector_size, output_vector_size])

    # this is standard creation of RNN cell, approximately equivalent to:
    # def rnn_cell(rnn_input, state):
    #    with tf.variable_scope('rnn_cell', reuse=True):
    #        W = tf.get_variable('W', [num_classes + state_size, state_size])
    #        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    #    return tf.relu(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
    # this is described here:
    # https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
    layers = [tf.contrib.rnn.BasicRNNCell(num_units=neuron_count, activation=tf.nn.relu) for layer in range(layer_count)]

    # Now we need to stack our layers on top of each other, here are more
    # information on this with pictures:
    # https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

    # We do dynamic training method, i.e. training continues even during test
    # phase, dynamic model updated also just once more during test data
    # processing.
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    # standard square loss
    loss = tf.reduce_mean(tf.square(outputs - y))

    # adam optimization algorithm is an extension to stochastic gradient descent
    # https://arxiv.org/abs/1412.6980
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    if rebuild_artifacts or not os.path.isfile(ARTIFACTS + ".index"):
        with tf.Session() as sess:
            if build_tensorboard_logs:
                writer = tf.summary.FileWriter('logs', sess.graph)
            init.run()
            for iteration in range(n_iterations):
                X_batch, y_batch = next_batch(batch_size, state_vector_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                if iteration % 100 == 0:
                    mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                    print(iteration, "\tLoss:", mse)

            saver.save(sess, ARTIFACTS)
            if build_tensorboard_logs:
                writer.close()

    # now, let's verify if we can do good prediction on dev make_ds
    load_dataset(TARGET_DATASET)
    bX, by = next_batch(1, state_vector_size)

    with tf.Session() as sess:
        saver.restore(sess, ARTIFACTS)
        X_new = bX
        y_pred = sess.run(outputs, feed_dict={X: X_new})

    plt.title("SF Fire Department Incidents", fontsize=14)
    plt.plot(t_instance[1:], by[0,:,0], "g-", markersize=2, label="Target")
    plt.plot(t_instance[1:], y_pred[0,:,0], "r-", markersize=1, label="Prediction")
    plt.legend(loc="upper left")
    plt.xlabel("Hour of week")
    plt.ylabel("Number of incidents")

    plt.show()

if __name__ == "__main__":
    reset_graph()
    train_predict()