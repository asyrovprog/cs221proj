import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv, os, random

rebuild_artifacts = False               # do training even if weight already saved
build_tensorboard_logs = False          # build logs for tensorboard
num_prediction_samples = 20             # number of "creative" prediction weeks for RMSE calculation
verbous = True                          # some logging
training_loss_step = 50                 # loss report step
display_training_loss_chart = True      # create learning curve chart
use_GRU_cell = False                    # use GRU Cell instead of Vanilla RNN
run_on_test_dataset = False             # this should be false during development

num_steps = 24 * 31       # number of time-steps
neuron_count = 98         # number of units in hidden state (same for all Vanilla RNN cells)
learning_rate = 0.00019   # learning rate found during optimization
num_iters = 800           # iteration count found during optimization
batch_size = 50
layer_count = 3           # number of rnn cells layers

ARTIFACTS = "../calcs/sf-fire-rnn-smoke"       # artifacts (weights) file(s) prefix
DATASET   = "../ds/sf-fire-counts-train.csv"   # training dataset
if run_on_test_dataset:
    TARGET_DATASET = "../ds/sf-fire-counts-test.csv"
else:
    TARGET_DATASET = "../ds/sf-fire-counts-dev.csv"
data = []

# load dataset
def load_dataset(filename):
    global data
    with open(filename, "rt") as f:
        reader = csv.DictReader(f)
        data = []
        for i, line in enumerate(reader):
            data.append(float(line["count"]))
        data = np.array(data)

# initialize tensorflow graph
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


# calculate root of mean of difference squared, arrays must be np.array()
def RMSE(predictions, targets):
    diff_squared = (predictions - targets) ** 2
    return np.sqrt(diff_squared.mean())

def train_predict():
    load_dataset(DATASET)
    X = tf.placeholder(tf.float32, [None, num_steps, 1])
    y = tf.placeholder(tf.float32, [None, num_steps, 1])

    if use_GRU_cell:
        layers = [tf.contrib.rnn.GRUCell(num_units=neuron_count, activation=tf.nn.relu) for layer in range(layer_count)]
    else:
        layers = [tf.contrib.rnn.BasicRNNCell(num_units=neuron_count, activation=tf.nn.relu) for layer in range(layer_count)]
    # Now we need to stack our layers on top of each other
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    if verbous:
        for i, l in enumerate(layers + [multi_layer_cell]):
            print("cell {}: state_size: {}, output_size: {}".format(str(i), l.state_size, l.output_size))

    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    if rebuild_artifacts or not os.path.isfile(ARTIFACTS + ".index"):
        losses = []
        with tf.Session() as sess:
            if build_tensorboard_logs:
                writer = tf.summary.FileWriter('logs', sess.graph)
            init.run()
            for iteration in range(num_iters):
                x_batch, y_batch, ids = next_batch(batch_size, num_steps)
                sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
                if iteration % training_loss_step == 0 or iteration == num_iters - 1:
                    mse = loss.eval(feed_dict={X: x_batch, y: y_batch})
                    losses.append(mse)
                    print(iteration, "\tLoss:", mse)

            saver.save(sess, ARTIFACTS)
            if build_tensorboard_logs:
                writer.close()

        if display_training_loss_chart:
            x = [i * training_loss_step for i in range(0, len(losses))]
            plt.title("Training curve", fontsize=14)
            plt.plot(x, losses, "g-", markersize=3)
            plt.xlabel("iteration", fontsize=14)
            plt.ylabel("Mean Square Loss", fontsize=14)
            plt.show()

    # now, let's verify if we can do good prediction on dev make_ds
    load_dataset(TARGET_DATASET)

    with tf.Session() as sess:
        saver.restore(sess, ARTIFACTS)
        all_targets, all_predictions = [], []
        for i in range(0, num_prediction_samples):
            bx, by, ids = next_batch(1, num_steps)
            targets, predictions = [], []
            sequence = [i for i in bx[0, :, 0]]
            for iteration in range(24 * 7):
                x_batch = np.array(sequence[-num_steps:]).reshape(1, num_steps, 1)
                y_pred = sess.run(outputs, feed_dict={X: x_batch})
                sequence.append(y_pred[0, -1, 0])
                predictions.append(y_pred[0, -1, 0])
                targets.append(data[ids[0] + len(sequence)])
            all_targets += targets
            all_predictions += predictions
            if verbous:
                print("RMSE #{}: {}".format(i, RMSE(np.array(targets), np.array(predictions))))

        final_rmse = RMSE(np.array(all_targets), np.array(all_predictions))
        print("Final RMSE: {}".format(final_rmse))

        x = [i for i in range(0, len(targets))]
        plt.title("SF Fire Department Incidents", fontsize=14)
        plt.plot(x, targets, "g-", markersize=2, label="Target")
        plt.plot(x, predictions, "r-", markersize=1, label="Prediction")
        plt.legend(loc="upper left")
        plt.xlabel("Hour")
        plt.ylabel("Number of incidents")
        plt.show()

if __name__ == "__main__":
    reset_graph()
    train_predict()