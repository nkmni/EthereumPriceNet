import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SCRIPT_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_REL_PATH = 'data/'
RESULTS_DIR_REL_PATH = 'results/'
DATA_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, DATA_DIR_REL_PATH)
RESULTS_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, RESULTS_DIR_REL_PATH)


def load_data():
    X_train = np.load(os.path.join(DATA_DIR_ABS_PATH, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR_ABS_PATH, 'y_train.npy'))
    X_dev = np.load(os.path.join(DATA_DIR_ABS_PATH, 'X_dev.npy'))
    y_dev = np.load(os.path.join(DATA_DIR_ABS_PATH, 'y_dev.npy'))
    # X_test = np.load(os.path.join(DATA_DIR_ABS_PATH, 'X_test.npy'))
    # y_test = np.load(os.path.join(DATA_DIR_ABS_PATH, 'y_test.npy'))
    # return (X_train, y_train, X_dev, y_dev, X_test, y_test)
    return (X_train, y_train, X_dev, y_dev)


def generate_model():
    inputs = tf.keras.Input(shape=(14, 20))
    #batch_norm = tf.keras.layers.BatchNormalization()(inputs)
    lstm1 = tf.keras.layers.LSTM(units=256, return_sequences=True)(inputs)
    lstm2 = tf.keras.layers.LSTM(units=256)(lstm1)
    #batch_norm2 = tf.keras.layers.BatchNormalization()(lstm2)
    dense = tf.keras.layers.Dense(units=256)(lstm2)
    leaky_relu = tf.keras.layers.LeakyReLU()(dense)
    outputs = tf.keras.layers.Dense(units=3)(leaky_relu)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def draw_results(y, predictions, title):
    plt.figure()
    plt.plot(y[:, 0], label='Min (actual)')
    plt.plot(y[:, 1], label='Avg (actual)')
    plt.plot(y[:, 2], label='Max (actual)')
    plt.plot(predictions[:, 0], label='Min (predicted)')
    plt.plot(predictions[:, 1], label='Min (predicted)')
    plt.plot(predictions[:, 2], label='Min (predicted)')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, title + '.png'), dpi=300, format='png')


def draw_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('History')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, 'history.png'), dpi=300, format='png')


if __name__ == '__main__':
    #X_train, y_train, X_dev, y_dev, X_test, y_test = load_data()
    X_train, y_train, X_dev, y_dev = load_data()
    model = generate_model()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_dev, y_dev))
    #model.evaluate(X_test, y_test)
    train_predictions = model.predict(X_train)
    draw_results(y_train, train_predictions, 'train')
    dev_predictions = model.predict(X_dev)
    draw_results(y_dev, dev_predictions, 'dev')
    draw_history(history)
    print(history)
