import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv, os, random

rebuild_artifacts = True
build_tensorboard_logs = False
num_prediction_samples = 1 # 20
verbous = True
training_loss_step = 50
display_training_loss_chart = True

# number of previous values
num_steps = 24 * 31
# we have just one input, which is number of emergency phone calls during current hour
input_size = 1
# number of units in hidden state (same for all Vanilla RNN cells)
neuron_count = 98
# our prediction is again number of phone calls for hour
output_size = 1
# learning rate found during optimization in dev phase
learning_rate = 0.00019
# iteration count found during optimization in dev phase
num_iters = 800
# iteration count found during optimization in dev phase
batch_size = 50
# number of layers found during optimization in dev phase
layer_count = 3


ARTIFACTS = "../calcs/sf-fire-rnn-smoke"
DATASET = "../ds/sf-fire-counts-train.csv"
TARGET_DATASET = "../ds/sf-fire-counts-test.csv"
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


# calculate root of mean of difference squared, arrays must be np.array()
def RMSE(predictions, targets):
    diff = predictions - targets
    diff_squared = diff ** 2
    mean_of_diff = diff_squared.mean()
    rmse_val = np.sqrt(mean_of_diff)
    return rmse_val

def train_predict():
    load_dataset(DATASET)

    X = tf.placeholder(tf.float32, [None, num_steps, input_size])
    y = tf.placeholder(tf.float32, [None, num_steps, output_size])

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
                X_batch, y_batch, ids = next_batch(batch_size, num_steps)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                if iteration % training_loss_step == 0 or iteration == num_iters - 1:
                    mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
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
            bX, by, ids = next_batch(1, num_steps)
            targets, predictions = [], []
            sequence = [i for i in bX[0, :, 0]]
            for iteration in range(24 * 7):
                X_batch = np.array(sequence[-num_steps:]).reshape(1, num_steps, 1)
                y_pred = sess.run(outputs, feed_dict={X: X_batch})
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