import sys
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras.optimizers import Adam, AdamW

sys.path.append("..")  # Adds higher directory to python modules path. So you can import parent.pa
from train_wc_nonpsyns import WilsonCowanNetwork
import config


pconn = np.zeros((config.N_INPUTS + config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float32 )

pconn[0, 0] = 1.0
pconn[-1, 0] = 1.0

e_r = np.zeros((config.N_INPUTS + config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float32 ) + 1.0
e_r[0, 1] = -1.0

params = {

    "pconn" : pconn,
    "e_r" : e_r,
    "tau_pop" :  np.zeros((config.N_POP_UNITS, ), dtype=np.float32 ) + 10,
    'I_ext' : np.zeros((config.N_POP_UNITS, ), dtype=np.float32 ) + 5.0,
    "gsyn_max" : np.zeros((config.N_INPUTS + config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float32 ) + 10.0,
    "tau_1" : np.zeros((config.N_INPUTS + config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float32 ) + 5.0,
    "tau_2" : np.zeros((config.N_INPUTS + config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float32 ) + 15.0,

}

print(params["pconn"])

batch_size = 2

inputs = Input(shape=(None, config.N_INPUTS), batch_size=batch_size)

cell = WilsonCowanNetwork(params, dt=0.1, batch_size=batch_size)

x = RNN(cell, return_sequences=True, stateful=True, name='wc_rnn')(inputs)



model = Model(inputs, x)

def loss_with_reg(y_true, y_pred):
    L_mse = tf.keras.losses.MSLE(y_true, y_pred[..., :4])  #      #tf.keras.losses.MeanSquaredError()(y_true, y_pred[..., :4])  #      #tf.keras.losses.MeanSquaredError()(y_true, y_pred[..., :4])  #      #tf.keras.losses
    return L_mse



model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss=loss_with_reg,
)


X = np.zeros((batch_size, 1000, config.N_INPUTS ), dtype=np.float32) + 10.0

Ypred = model.predict(X)


plt.plot(Ypred[0, :, 0])
plt.plot(Ypred[0, :, 1])
plt.plot(Ypred[0, :, 3])
plt.show()

