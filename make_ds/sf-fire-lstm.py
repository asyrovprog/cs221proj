import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv, os, random, math
n_steps = 24
n_inputs = 1
n_neurons = 48
n_outputs = 1
learning_rate = 0.0001
n_iterations = 1500
batch_size = 50
n_layers = 3
normalization_factor = 200
n_test_iters = 100

ARTIFACTS = "../calcs/{}/sf-fire"
DATASET = "../ds/sf-fire-counts-train.csv"
TARGET_DATASET = "../ds/sf-fire-counts-dev.csv"

data = []
t_instance = np.linspace(0, n_steps + 1, n_steps + 1)

def normalize(x):
    return x / normalization_factor

def denormalize(x):
    return x * normalization_factor

def load_dataset(filename):
    global data
    with open(filename, "rt") as f:
        reader = csv.DictReader(f)
        data = []
        for i, line in enumerate(reader):
            data.append(normalize(float(line["count"])))
        data = np.array(data)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def next_batch(n_batch, n_steps):
    count = len(data)
    ys = []
    for b in range(0, n_batch):
        i = random.randint(0, count - n_steps * 2 - 1)
        run = []
        for s in range(0, n_steps * 2 + 1):
            run.append(data[i])
            i += 1
        ys.append(run)
    ys = np.array(ys)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

def get_rnn(X):
    layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
              for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    return outputs

def get_lstm(X):
    layers = [tf.contrib.rnn.LSTMCell(num_units=800, forget_bias=0.2)
              for layer in range(n_layers)]
    result_cell = tf.contrib.rnn.MultiRNNCell(layers)
    #result_cell = tf.contrib.rnn.LSTMCell(num_units=800, forget_bias=0.2)

    outputs, states = tf.nn.dynamic_rnn(result_cell, X, dtype=tf.float32)
    return outputs

def get_loss(outputs, y):
    return tf.reduce_mean(tf.square(outputs - y))

def train_and_save(name, model_func, X, y):
    outputs = model_func(X)
    loss = get_loss(outputs, y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    if not os.path.isfile(ARTIFACTS.format(name) + ".index"):
        with tf.Session() as sess:
            init.run()
            for iteration in range(n_iterations):
                X_batch, y_batch = next_batch(batch_size, n_steps)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                if iteration % 10 == 0:
                    mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                    print(name, "\t", iteration, "\tLoss:", mse)
            saver.save(sess, ARTIFACTS.format(name))

    return outputs

def get_prediction(name, outputs, X, bX):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ARTIFACTS.format(name))
        X_new = bX
        y_pred = sess.run(outputs, feed_dict={X: X_new})
    return y_pred[0,:,0]

def get_test_rmse(name, output, X):
    samples = []
    for i in range(n_test_iters):
        bX, by = next_batch(1, n_steps)
        prediction = get_prediction(name, output, X, bX)
        mse = ((by[0,:,0]-prediction)**2).mean()
        samples.append(math.sqrt(mse))
    return sum(samples)/len(samples)

def train_predict():
    load_dataset(DATASET)

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    models = {}
    #models['lstm'] = train_and_save('lstm', get_lstm, X, y)
    models['rnn'] = train_and_save('rnn', get_rnn, X, y)
    
    load_dataset(TARGET_DATASET)

    for name in models:
        print(name, "\t", "test rmse:", get_test_rmse(name, models[name], X))
    
    
    bX, by = next_batch(1, n_steps)
    target = denormalize(by[0,:,0])
    predictions = {name: denormalize(get_prediction(name, output, X, bX))
                   for name, output in models.items()}

    plt.title("SF Fire Department Incidents", fontsize=14)
    plt.plot(t_instance[1:], target, "g-", markersize=2, label="Target")
    for name in predictions:
        plt.plot(t_instance[1:], predictions[name], "b-" if name=='lstm' else "r-", markersize=1, label="Prediction " + name)
    plt.legend(loc="upper left")
    plt.xlabel("Hours")

    plt.show()

if __name__ == "__main__":
    reset_graph()
    train_predict()
