import os
import tensorflow as tf
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt

import keras 

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
          

plt.close('all')
plt.ioff()

"""
Script for creating an ML model in order to do regression on a nonlinear model.
We added gradient data and minimize the MSE.
"""

# Initial 
n_samples   = 100
lower,upper = -1,1 

# Training values and gradients
def fun(x):
    return np.power(x, 5) + 0.4 * x 
def dfun(x):
    return 5*np.power(x, 4) + 0.4

x_train         = np.random.rand(n_samples) * (upper-lower) + lower
y_train         = fun(x_train)
y_grad_train    = dfun(x_train)

# Real data for plotting
xreal  = np.linspace(-1, 1, 500)[:, np.newaxis]
yreal  = fun(xreal)
dyreal = dfun(xreal)

model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(1,)),
        tf.keras.layers.Dense(10, activation="sigmoid"),
        tf.keras.layers.Dense(10, activation="sigmoid"),
        tf.keras.layers.Dense(10, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="linear")
    ]
    )

# Instantiate an optimizer.
optimiser = keras.optimizers.Adam(learning_rate=1e-4)

x_train         = tf.convert_to_tensor(x_train)
y_grad_train    = tf.convert_to_tensor(y_grad_train)
y_train         = tf.convert_to_tensor(y_train)

@tf.function
def train_step(x, y, y_grad):

    y_hat = model(x)
    y_grad_hat = tf.gradients(y_hat, x)

    data_loss  = tf.math.reduce_mean(tf.math.square(y - y_hat))
    grad_loss  = tf.math.reduce_mean(tf.math.square(y_grad - y_grad_hat))

    # Total loss 
    loss =  data_loss +  grad_loss

    # Minimise the combined loss with respect to the neural network
    gradients = tf.gradients(loss, model.trainable_weights)
    optimiser.apply_gradients(zip(gradients, model.trainable_weights))
    return loss

# Train model
epochs = 12000
for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))
    loss_value = train_step(x_train ,y_train ,y_grad_train)
    print("Training loss at epoch %d: %.4f"% (epoch, float(loss_value)))

# Predict values
y_pred = model.predict(xreal)

# Plot everything
plt.figure()
plt.plot(xreal,y_pred,"b--",zorder = 2,label = "Predicted function")
plt.plot(xreal,yreal,"r--",zorder = 2,label = "Real function")
plt.scatter(x_train, y_train);
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
