import os
import tensorflow as tf
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
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
n_samples   = int(20)
lower,upper = -1,1 
dim = 1

# Training values and gradients
def fun(x):
    return np.power(x, 5) + 0.4 * x 
def dfun(x):
    return 5*np.power(x, 4) + 0.4

x_train  = np.random.rand(n_samples) * (upper-lower) + lower
y_train  = fun(x_train)
dy_train = dfun(x_train)

y_train = np.concatenate([np.atleast_2d(y_train),np.atleast_2d(dy_train)]).T

# Real data for plotting
xreal = np.linspace(-1, 1, 500)[:, np.newaxis]
yreal = fun(xreal)
dyreal = dfun(xreal)

# Extended loss function
def extendedmse(y_true, y_pred):
    squared_difference = tf.square(y_true[:,0] - y_pred[:,0])
    squared_differencegrad = tf.square(y_true[:,1] - y_pred[:,1])       
    return tf.reduce_mean(squared_difference) + tf.reduce_mean(squared_differencegrad) 


def create_model():
    # ML Model
    inputlayer = Input(shape=[1],name="input")
    hidden = Dense(20, activation="sigmoid")(inputlayer)
    hidden = Dense(20, activation="sigmoid")(hidden)
    
    # Value path
    y_1 = keras.layers.Dense(1,name="y_1")(hidden)
    
    # Gradient path
    y_2 = keras.layers.Dense(dim,name="y_2")(hidden)
    hiddengrad = keras.layers.Dense(20, activation="sigmoid")(y_2)
    hiddengrad = keras.layers.Dense(20, activation="sigmoid")(hiddengrad)
    ygrad = keras.layers.Dense(dim,name="ygrad")(hiddengrad)
    
    # Output layer
    output = keras.layers.concatenate([y_1,ygrad])
    
    # Build and compile
    model = keras.models.Model(inputs=inputlayer,outputs=output)
    model.compile(Adam(learning_rate=0.01), loss=extendedmse, run_eagerly=True) 
    
    return model


def main():

    model = create_model()
    model.summary()
    
    # Fit model to data
    history = model.fit(x_train, y_train, epochs=2000, verbose=1)
    
    # Prediction
    y_hat = model.predict(xreal)
    
    #Plotting 
    plt.figure(0)
    lossiter = np.arange(len(history.history['loss']))
    plt.plot(lossiter,history.history['loss'],"b--")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale("log")
    
    plt.figure(1)
    plt.plot(xreal,y_hat[:,1],"b--",zorder = 2,label = "Predicted gradient")
    plt.plot(xreal,dyreal,"r--",zorder = 2,label = "Real gradient")
    #plt.scatter(x_train, y_train[:,0]);
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    plt.figure(2)
    plt.plot(xreal,y_hat[:,0],"b--",zorder = 2,label = "Predicted function")
    plt.plot(xreal,yreal,"r--",zorder = 2,label = "Real function")
    plt.scatter(x_train, y_train[:,0]);
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()