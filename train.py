import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utilities import Hyperparameters

SCRIPT_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_REL_PATH = 'data/'
RESULTS_DIR_REL_PATH = 'results/' + datetime.now().isoformat(' ', 'seconds') + '/'
DATA_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, DATA_DIR_REL_PATH)
RESULTS_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, RESULTS_DIR_REL_PATH)


def load_data():
    X = np.load(os.path.join(DATA_DIR_ABS_PATH, 'X.npy'))
    X_prices = np.load(os.path.join(DATA_DIR_ABS_PATH, 'X_prices.npy'))
    y = np.load(os.path.join(DATA_DIR_ABS_PATH, 'y.npy'))
    return (X, X_prices, y)


def generate_model(hps):
    lstm_inputs = tf.keras.Input(shape=(hps.sequence_length, 6 + hps.fft_window_size), name='lstm_inputs')
    lstm1 = tf.keras.layers.LSTM(units=hps.lstm1_units, return_sequences=True)(lstm_inputs)
    lstm2 = tf.keras.layers.LSTM(units=hps.lstm2_units)(lstm1)

    prices = tf.keras.Input(shape=(hps.sequence_length,), name='prices')
    concat = tf.keras.layers.concatenate([lstm2, prices])
    dense = tf.keras.layers.Dense(units=hps.dense_units)(concat)
    leaky_relu = tf.keras.layers.LeakyReLU()(dense)
    outputs = tf.keras.layers.Dense(units=3)(leaky_relu)

    model = tf.keras.Model(inputs=[lstm_inputs, prices], outputs=outputs)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=hps.learning_rate), metrics=['mae'])
    return model


def draw_results(y, predictions, title):
    plt.figure()
    plt.plot(y[:, 0], label='min (actual)')
    plt.plot(y[:, 1], label='avg (actual)')
    plt.plot(y[:, 2], label='max (actual)')
    plt.plot(predictions[:, 0], label='min (predicted)')
    plt.plot(predictions[:, 1], label='avg (predicted)')
    plt.plot(predictions[:, 2], label='max (predicted)')
    plt.xlabel('day')
    plt.ylabel('price')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, title + '.png'), dpi=600, format='png')


def draw_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (mse)')
    plt.title('history')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, 'history.png'), dpi=600, format='png')


def split_data(hps, X, X_prices, y):
    index = int(hps.train_split * X.shape[0])
    return (X[:index], X_prices[:index], y[:index], X[index:], X_prices[index:], y[index:])


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR_ABS_PATH)
    hps = Hyperparameters()
    hps.save(os.path.join(RESULTS_DIR_ABS_PATH, 'hyperparameters.json'))

    with open(os.path.join(RESULTS_DIR_ABS_PATH, 'output.txt'), 'w') as f:
        X, X_prices, y = load_data()
        X_train, X_train_prices, y_train, X_val, X_val_prices, y_val = split_data(hps, X, X_prices, y)

        model = generate_model(hps)
        tf.keras.utils.plot_model(model, os.path.join(RESULTS_DIR_ABS_PATH, 'model.png'), show_shapes=True)
        model.summary()

        if (len(sys.argv) > 1):
            model.load_weights(sys.argv[1])

        history = model.fit(
            {'lstm_inputs': X_train, 'prices': X_train_prices},
            y_train,
            epochs=hps.epochs,
            batch_size=hps.batch_size,
            validation_data=({'lstm_inputs': X_val, 'prices': X_val_prices}, y_val)
        )

        model.save_weights(os.path.join(RESULTS_DIR_ABS_PATH, 'weights.h5'))

        train_predictions = model.predict([X_train, X_train_prices])
        val_predictions = model.predict([X_val, X_val_prices])

        draw_results(y_train, train_predictions, 'train')
        draw_results(y_val, val_predictions, 'validation')
        draw_history(history)

        f.write('    loss: {}\n     mae: {}\nval loss: {}\n val mae: {}\n'.format(
            history.history['loss'][-1],
            history.history['mae'][-1],
            history.history['val_loss'][-1],
            history.history['val_mae'][-1]
        ))
        f.write(' val rho: {}'.format(np.corrcoef(y_val, val_predictions)[0, 1]))
