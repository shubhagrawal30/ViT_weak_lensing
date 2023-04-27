import numpy as np
from matplotlib import pyplot as plt

# a = np.load("../data/6-parameter-maps/cosmo_params_all_collage_flask_6cosmo_n625_jr0_wSN0_jz3.npz.npy", allow_pickle=True)
b = np.load("../data/6-parameter-maps/X_maps_Cosmogrid_100k.npy", allow_pickle=True)
# b = np.load("../data/6-parameter-maps/maps_all_collage_flask_6cosmo_n625_jr0_wSN0_jz3.npz.npy", allow_pickle=True)
a = np.load("../data/6-parameter-maps/y_maps_Cosmogrid_100k.npy", allow_pickle=True)

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from datetime import datetime
import tensorflow_probability as tfp
tfd = tfp.distributions

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, PROGRESS_EPOCH=50):
        self.PROGRESS_EPOCH = PROGRESS_EPOCH
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.PROGRESS_EPOCH == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')}, epoch {epoch}: ", end="")
            for key, val in logs.items():
                print(f"{key}: {val:.3f}", end= "\t")
            print()

tf.config.list_physical_devices('GPU')

num_samples = len(a)
train_split, val_split, test_split = int(0.80*num_samples), \
            int(0.10*num_samples), int(0.10*num_samples) + 1

# print(train_split, val_split, test_split, train_split+val_split+test_split)

train_x, val_x, test_x = np.split(b, [train_split, train_split+val_split])
train_y, val_y, test_y = np.split(a, [train_split, train_split+val_split])

# print(train_x.shape, val_x.shape, test_x.shape)
# print(train_y.shape, val_y.shape, test_y.shape)
# print(train_y.shape, val_y.shape, test_y.shape)

output_num = 5

train_y, val_y, test_y = train_y[:,:output_num], val_y[:,:output_num], test_y[:,:output_num]

from sklearn.preprocessing import StandardScaler
scaler_y = StandardScaler()
scaler_y.fit(a[:, :5])

# print(np.mean(train_y, axis=0))
# print(np.mean(val_y, axis=0))
# print(np.mean(test_y, axis=0))

train_y, val_y, test_y = scaler_y.transform(train_y), \
        scaler_y.transform(val_y), scaler_y.transform(test_y) 


# print(np.mean(train_y, axis=0))
# print(np.mean(val_y, axis=0))
# print(np.mean(test_y, axis=0))
# print(np.std(train_y, axis=0))
# print(np.std(val_y, axis=0))
# print(np.std(test_y, axis=0))

input_shape = (66, 66, 1)

tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Normalization(),
  tf.keras.layers.Conv2D(8, kernel_size=3, activation='tanh'),
  tf.keras.layers.AveragePooling2D(pool_size=2),
  tf.keras.layers.Conv2D(16, kernel_size=3, activation='tanh'),
  tf.keras.layers.AveragePooling2D(pool_size=2),
  tf.keras.layers.Conv2D(16, kernel_size=(2, 2), activation='tanh'),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(32, kernel_size=2, activation='tanh'),
  tf.keras.layers.AveragePooling2D(pool_size=2),
  tf.keras.layers.Conv2D(32, kernel_size=2, activation='tanh'),
  tf.keras.layers.AveragePooling2D(pool_size=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='tanh'),
  tf.keras.layers.Dense(256, activation='tanh'),
  tf.keras.layers.Dense(128, activation='tanh'),
  tf.keras.layers.Dense(64, activation='tanh'),
  tf.keras.layers.Dense(output_num, activation='linear') # assuming 6 output parameters
])

# Compile the model
# model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
model.compile(loss=tf.keras.losses.LogCosh(), optimizer=tf.keras.optimizers.Adam())

model.summary()

model.fit(train_x, train_y, epochs=10, verbose=0, callbacks=[CustomCallback(2)], validation_data=(val_x, val_y))

plot_x, plot_y = train_x, scaler_y.inverse_transform(train_y)
predictions = scaler_y.inverse_transform(model.predict(plot_x))
upp_lims = np.nanmax(plot_y, axis=0)
low_lims = np.nanmin(plot_y, axis=0)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
fig.subplots_adjust(wspace=0.3, hspace=0.2)
labels = [r"$\Omega_m$", r"$H_0$", r"$n_s$", r"$\sigma_8$", r"$w_0$", r"$$"]
for ind, (label, ax, low_lim, upp_lim) in enumerate(zip(labels, axs.ravel(), low_lims, upp_lims)):
    p = np.poly1d(np.polyfit(plot_y[:, ind], predictions[:, ind], 1))
    ax.scatter(plot_y[:, ind], predictions[:, ind], marker="x", alpha=0.1)
    ax.set_xlabel("true")
    ax.set_ylabel("prediction")
    ax.plot([low_lim, upp_lim], [low_lim, upp_lim], color="black")
    ax.plot([low_lim, upp_lim], [p(low_lim), p(upp_lim)], color="black", ls=":")
    ax.set_xlim([low_lim, upp_lim])
    ax.set_ylim([low_lim, upp_lim])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(label)
    ax.grid()
plt.savefig("train.png")
plt.close()
    
plot_x, plot_y = val_x, scaler_y.inverse_transform(val_y)
predictions = scaler_y.inverse_transform(model.predict(plot_x))
upp_lims = np.nanmax(plot_y, axis=0)
low_lims = np.nanmin(plot_y, axis=0)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
fig.subplots_adjust(wspace=0.3, hspace=0.2)
for ind, (label, ax, low_lim, upp_lim) in enumerate(zip(labels, axs.ravel(), low_lims, upp_lims)):
    p = np.poly1d(np.polyfit(plot_y[:, ind], predictions[:, ind], 1))
    ax.scatter(plot_y[:, ind], predictions[:, ind], marker="x", alpha=0.1)
    ax.set_xlabel("true")
    ax.set_ylabel("prediction")
    ax.plot([low_lim, upp_lim], [low_lim, upp_lim], color="black")
    ax.plot([low_lim, upp_lim], [p(low_lim), p(upp_lim)], color="black", ls=":")
    ax.set_xlim([low_lim, upp_lim])
    ax.set_ylim([low_lim, upp_lim])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(label)
    ax.grid()
plt.savefig("val.png")
plt.close()