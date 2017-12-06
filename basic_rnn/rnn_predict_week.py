import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv, os, random

rebuild_artifacts = True
build_tensorboard_logs = False

# number of previous values
num_steps = 24 * 12

# we have just one input, which is number of emergency phone calls during current hour
input_size = 1

# number of steps and size of prediction vector (prev + input_size + bias)
neuron_count = num_steps + input_size + 1

# our prediction is again number of phone calls for hour
output_size = 1

# learning rate found during optimization in dev phase
learning_rate = 0.0005

# iteration count found during optimization in dev phase
num_iters = 600

# iteration count found during optimization in dev phase
batch_size = 50

# number of layers found during optimization in dev phase
layer_count = 3

ARTIFACTS = "../calcs/sf-fire-rnn-smoke"
DATASET = "../ds/sf-fire-counts-train.csv"
TARGET_DATASET = "../ds/sf-fire-counts-dev.csv"
data = []

def load_dataset(filename):
    global data
    with open(filename, "rt") as f:
        reader = csv.DictReader(f)
        data = []
        for i, line in enumerate(reader):
            data.append(float(line["count"]))
        data = np.array(data)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def next_batch(n_batch, n_steps):
    count = len(data)
    ys, ids = [], []
    for b in range(0, n_batch):
        i = random.randint(0, count - n_steps - 1)
        ids.append(i)
        run = []
        for k in range(i, i + n_steps + 1):
            run.append(data[k])
        ys.append(run)
    ys = np.array(ys)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1), ids


def train_predict():
    load_dataset(DATASET)

    X = tf.placeholder(tf.float32, [None, num_steps, input_size])
    y = tf.placeholder(tf.float32, [None, num_steps, output_size])

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

    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    if rebuild_artifacts or not os.path.isfile(ARTIFACTS + ".index"):
        with tf.Session() as sess:
            if build_tensorboard_logs:
                writer = tf.summary.FileWriter('logs', sess.graph)
            init.run()
            for iteration in range(num_iters):
                X_batch, y_batch, ids = next_batch(batch_size, num_steps)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                if iteration % 100 == 0:
                    mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                    print(iteration, "\tLoss:", mse)

            saver.save(sess, ARTIFACTS)
            if build_tensorboard_logs:
                writer.close()

    # now, let's verify if we can do good prediction on dev make_ds
    load_dataset(TARGET_DATASET)

    with tf.Session() as sess:
        bX, by, ids = next_batch(1, num_steps)
        saver.restore(sess, ARTIFACTS)
        # X_new = bX
        # y_pred = sess.run(outputs, feed_dict={X: X_new})
        # plt.title("SF Fire Department Incidents", fontsize=14)
        # plt.plot(t_instance[1:], by[0,:,0], "g-", markersize=2, label="Target")
        # plt.plot(t_instance[1:], y_pred[0,:,0], "r-", markersize=1, label="Prediction")
        # plt.legend(loc="upper left")
        # plt.xlabel("Hour of week")
        # plt.ylabel("Number of incidents")
        # plt.show()
        # Result: achieved rMSE = 5.97 % on 3 - layer RNN with 50 neurons

        targets, predictions = [], []
        sequence = [i for i in bX[0, :, 0]]
        for iteration in range(24 * 7):
            X_batch = np.array(sequence[-num_steps:]).reshape(1, num_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            sequence.append(y_pred[0, -1, 0])
            predictions.append(y_pred[0, -1, 0])
            targets.append(data[ids[0] + len(sequence)])

        x = [i for i in range(0, len(targets))]
        plt.title("SF Fire Department Incidents", fontsize=14)
        plt.plot(x, targets, "g-", markersize=2, label="Target")
        plt.plot(x, predictions, "r-", markersize=1, label="Prediction")
        plt.legend(loc="upper left")
        plt.xlabel("Hour")
        plt.ylabel("Number of incidents")
        plt.show()

        # https://en.wikipedia.org/wiki/Root-mean-square_deviation

if __name__ == "__main__":
    reset_graph()
    train_predict()