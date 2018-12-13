# coding=utf-8
# @author: cer
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys


class AutoEncoder:
    def __init__(self, model):
        self.model = model
        self.vocab_size = model.vocab_size
        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        self.build()

    def tf_init(self, sess):
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        sess.run(init_new_vars_op)

    def build(self):
        input_layer = self.model.encoder_final_state_h
        self.output_true = input_layer
        output_len = input_layer.shape[1].value
        self.mid_layer_size = 32
        
        n_input = output_len
        n_hidden_1 = self.mid_layer_size
        n_hidden_2 = self.mid_layer_size
        with tf.variable_scope("ae_vars"):
            self.weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),                    
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),                    
                'out': tf.Variable(tf.random_normal([n_hidden_2, output_len]))
            }
            self.biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'out': tf.Variable(tf.random_normal([output_len]))
            }
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, self.weights['h1']), self.biases['b1']))        
        # layer_1 = tf.nn.dropout(layer_1, 0.8)
        
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        # layer_2 = layer_1
        # layer_2 = tf.nn.dropout(layer_2, 0.8)

        layer_3 = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])

        self.output_layer = layer_3
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ae_vars')        

        # multiply output of input_layer wth a weight matrix and add biases
        # layer_1 = tf.layers.dense(inputs=input_layer, units=self.mid_layer_size,
        #     activation=tf.nn.sigmoid, use_bias=True, name='ae_layer_1')
        # layer_2 = tf.layers.dense(inputs=layer_1, units=self.mid_layer_size,
        #     activation=tf.nn.sigmoid, use_bias=True, name='ae_layer_2')
        # self.output_layer = tf.layers.dense(inputs=layer_2, units=output_len,
        #     activation=None, use_bias=True, name='ae_layer_out')
        # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nlu_model/ae_layer_1')
        # train_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nlu_model/ae_layer_2'))
        # train_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nlu_model/ae_layer_out'))
        
        
        # define our cost function        
        # lossVec = tf.subtract(self.output_layer, self.output_true)
        # self.loss = tf.nn.l2_loss(lossVec)
        self.loss_per_col = tf.reduce_mean(tf.pow(self.output_layer - input_layer, 2), 1)
        self.loss = tf.reduce_mean(self.loss_per_col)
        
        # define our optimizer
        learn_rate = 0.01   # how fast the model should learn                
        # optimizer = tf.train.AdagradOptimizer(learn_rate)
        
        # optimizer.minimize(self.loss, var_list = train_vars)
        # optimizer.minimize(self.loss)
        with tf.name_scope("adam_optimizer_ae"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate, name="a_optimizer_ae")
            grads, vars = zip(*optimizer.compute_gradients(self.loss, var_list=train_vars))            
            # gradients, _ = tf.clip_by_global_norm(grads, 5)  # clip gradients
            gradients = grads  # clip gradients
            self.train_op = optimizer.apply_gradients(zip(gradients, vars), global_step=self.global_step)

    def step(self, sess, mode, trarin_batch):
        output_feeds, feed_dict = self.model.get_output_feed('test', trarin_batch)
        if (mode == 'train'):
            output_feeds.append(self.train_op)
        output_feeds.append(self.output_true)
        output_feeds.append(self.output_layer)
        output_feeds.append(self.loss)
        output_feeds.append(self.loss_per_col)
        return sess.run(output_feeds, feed_dict=feed_dict)
