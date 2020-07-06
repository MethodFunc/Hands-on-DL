import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow import keras


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    # Sign Curve 1
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    # Sign Curve 2
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    # Noise
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)

    return series[..., np.newaxis].astype(np.float32)


n_steps = 50
series = generate_time_series(10000, n_steps + 1)
x_train, y_train = series[:7000, :n_steps], series[:7000, -1]
x_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
x_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print(x_train.shape, x_valid.shape, x_test.shape)


def plot_series(series, y=None, y_pred=None, x_label='$t$', y_label='$x(t)$'):
    plt.plot(series, '.-')
    if y is not None:
        plt.plot(n_steps, y, 'bx', markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, 'ro')
    plt.grid(True)

    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)

    plt.hlines(0, 0, 100, lw=1)
    plt.axis([0, n_steps + 1, -1, 1])


fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
for col in range(3):
    plt.sca(axes[col])
    plot_series(x_valid[col, :, 0], y_valid[col, 0], y_label=('$x(t)$' if col == 0 else None))
plt.show()

y_pred = x_valid[:, -1]

print(np.mean(keras.losses.mean_squared_error(y_valid, y_pred)))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))

print(model.evaluate(x_test, y_test))
