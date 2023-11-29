#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:25:45 2023

@author: sourav
"""

import numpy as np 
import keras.backend as K
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow.keras.layers import  Dense
import logging
import os
from pathlib import Path


tf.config.run_functions_eagerly(True)

CHECKPOINTS_PATH = os.path.join('model')

class PgNN(tf.keras.Model):
    
    def __init__(self):

        super().__init__()
        self.checkpoints_dir = CHECKPOINTS_PATH
        self.model = tf.keras.Sequential()
        
        self.model.add(Dense(64, input_dim = 1, kernel_initializer = 'he_uniform', activation='relu' ))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1))
        self.optimizer = None
        self.loss_object = tf.keras.losses.MeanSquaredError()
        
        self.train_loss_results = {}
        self.train_accuracy_results = {}
        self.train_pred_results = {}
        self.w = 1e-04
        
    def Physics_Loss(self, v, v_pred, constraints):

        v_pred = v_pred[:,0]
        data_loss = K.square(v - v_pred) 
        
        c0 = constraints[:,0]
        c1 = constraints[:,1]
        c2 = constraints[:,2]
        c3 = constraints[:,3]
        c4 = constraints[:,4]
        
        #The index of coefficients is in the reverse order of what is written in the paper
        
        physics_loss = c4 + 2*c3*v_pred + 3*c2*v_pred**2 + 4*c1*v_pred**3 + 5*c0*v_pred**4

        return data_loss + self.w * physics_loss
    
    def summary(self):
        self.model.summary(print_fn=lambda x: logging.info(x))
        
    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)
        
    @tf.function  
    def train_step(self, data):

        depth, v, constraints = data

        with tf.GradientTape() as tape:
            v_pred = self.model(depth, training=True)  # Forward pass
            loss = self.Physics_Loss(v, v_pred, constraints)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss
    
    def load_weights(self, path=None):
        if path is None:
            path = self.checkpoints_dir

        self.model.load_weights(tf.train.latest_checkpoint(path))
        logging.info(f'\tWeights loaded from {path}')
        
    def save_weights(self, path):

        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path)
        
    def fit(self, depth, vel, constraints, epochs = None, depth_test=None, vel_test=None, optimizer='adam', learning_rate=0.01, verbose=1):
        depth = self.tensor(depth)
        vel = self.tensor(vel)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')

        for epoch in range(1, epochs + 1):
            loss = self.train_step([depth, vel, constraints])
            self.epoch_loss.update_state(loss)  # Add current batch loss
            self.epoch_callback(epoch, self.epoch_loss.result(), epochs, depth_test, vel_test, verbose)   
            
    def epoch_callback(self, epoch, epoch_loss, epochs, depth_val=None, v_val=None, verbose=1):


        self.train_loss_results[epoch] = epoch_loss
            
        length = len(str(epochs))
        log_str = f'\tEpoch: {str(epoch).zfill(length)}/{epochs},\t' \
                  f'Loss: {epoch_loss:.4e}'

        if depth_val is not None and v_val is not None:
            [mean_squared_error, errors, Y_pred] = self.evaluate(depth_val, v_val)
            self.train_accuracy_results[epoch] = mean_squared_error
            self.train_pred_results[epoch] = Y_pred
            log_str += f',\tAccuracy (MSE): {mean_squared_error:.4e}'
            if mean_squared_error <= min(self.train_accuracy_results.values()):
                self.save_weights(os.path.join(self.checkpoints_dir, 'easy_checkpoint'))
            logging.info(log_str)

        if epoch == epochs and depth_val is None and v_val is None:
            self.save_weights(os.path.join(self.checkpoints_dir, 'easy_checkpoint'))
            
        

    def evaluate(self, x_val, y_val, metric='MSE'):

        y_pred = self.model.predict(x_val)
        errors = None
        if metric == 'MSE':
            errors = tf.square(y_val - y_pred)
        elif metric == 'MAE':
            errors = tf.abs(y_val - y_pred)

        mean_error = tf.reduce_mean(errors)

        return mean_error, errors, y_pred
    
    
    def predict(self, x):

        return self.model.predict(x)
      

def get_model():
    model = PgNN()   
    model.compile(loss= PgNN.Physics_Loss, optimizer='adam')
    return model
    
def evaluate_model(x,y, constraints):
    cv = RepeatedKFold(n_splits = 10, n_repeats = 20, random_state = 1)
    for train_ix, test_ix in cv.split(X):
        depth_train, depth_test = x[train_ix], x[test_ix]
        v_train, v_test = y[train_ix], y[test_ix]
        c_train = constraints[train_ix]
        
    model = get_model()
    model.fit(depth = depth_train, vel = v_train, constraints = c_train, epochs = 2000, depth_test =  depth_test, vel_test = v_test, verbose = 1)
    

if __name__ == '__main__':

    dataset = np.load("data/dataset.npy", allow_pickle=True)
    _dict = dataset.item()
    X = np.array([*_dict.keys()])
    Y = np.array([*_dict.values()])[:,0].astype( "float64")
    
    
    constraints = np.array(np.array([*_dict.values()])[:,1].tolist())
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logging.info("TensorFlow version: {}".format(tf.version.VERSION))
    logging.info("Eager execution: {}".format(tf.executing_eagerly()))
    evaluate_model(X, Y, constraints)





