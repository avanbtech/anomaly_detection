# coding=utf-8
# @author: cer
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys


class Model:
    def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, index_to_embedding, 
                 batch_size=16, n_layers=1):
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.epoch_num = epoch_num
        self.encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size],
                                             name='encoder_inputs')
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length')
        self.intent_targets = tf.placeholder(tf.int32, [batch_size],
                                             name='intent_targets')
        self.global_step = tf.Variable(0, name='global_step',trainable=False)

        self.use_word_embed = False
        if (index_to_embedding is not None):
            self.vocab_size = index_to_embedding.shape[0]
            self.embedding_size = index_to_embedding.shape[1]
            self.embeddings = tf.Variable(
                tf.constant(0.0, shape=index_to_embedding.shape),
                trainable=False,
                name="Embedding")
            self.use_word_embed = True
        

    def build(self):
        if (not self.use_word_embed):
            self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size],-0.1, 0.1), dtype=tf.float32, name="embedding")        
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.embeddings, ids=self.encoder_inputs)

        # Encoder

        # Use a single LSTM cell
        encoder_f_cell_0 = LSTMCell(self.hidden_size)
        encoder_b_cell_0 = LSTMCell(self.hidden_size)
        encoder_f_cell = DropoutWrapper(encoder_f_cell_0,output_keep_prob=0.5)
        encoder_b_cell = DropoutWrapper(encoder_b_cell_0,output_keep_prob=0.5)        
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                            cell_bw=encoder_b_cell,
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32, time_major=True)
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        self.encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        self.encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=self.encoder_final_state_h
        )
        print("encoder_outputs: ", encoder_outputs)
        print("encoder_outputs[0]: ", encoder_outputs[0])
        print("encoder_final_state_c: ", encoder_final_state_c)

        intent_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.intent_size], -0.1, 0.1),
                               dtype=tf.float32, name="intent_W")
        intent_b = tf.Variable(tf.zeros([self.intent_size]), dtype=tf.float32, name="intent_b")

        # intent
        self.intent_logits = tf.add(tf.matmul(self.encoder_final_state_h, intent_W), intent_b)
        # intent_prob = tf.nn.softmax(intent_logits)
        self.intent = tf.argmax(self.intent_logits, axis=1)
        # intent loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.intent_targets, depth=self.intent_size, dtype=tf.float32),
            logits=self.intent_logits)
        loss_intent = tf.reduce_mean(cross_entropy)

        # self.loss = loss_slot + loss_intent
        self.loss = loss_intent
        optimizer = tf.train.AdamOptimizer(name="a_optimizer")
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        print("vars for loss function: ", self.vars)
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars), 
            global_step = self.global_step)

        # self.saver = tf.train.Saver(tf.global_variables())

    def get_output_feed(self, mode, train_batch):
        """ perform each batch"""
        if mode not in ['train', 'test']:
            print >> sys.stderr, 'mode is not supported'
            sys.exit(1)
        unziped = list(zip(*train_batch))
        # print(np.shape(unziped[0]), np.shape(unziped[1]),
        #       np.shape(unziped[2]), np.shape(unziped[3]))
        if mode == 'train':
            # output_feeds = [self.train_op, self.loss, self.decoder_prediction,self.intent, self.mask, self.slot_W]
            output_feeds = [self.train_op, self.loss, self.intent, self.intent_logits]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.encoder_inputs_actual_length: unziped[1],
                        #  self.decoder_targets: unziped[2],
                        #  self.intent_targets: unziped[3]}
                         self.intent_targets: unziped[2]}
        if mode in ['test']:
            # output_feeds = [self.decoder_prediction, self.intent]
            output_feeds = [self.intent, self.intent_logits]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.encoder_inputs_actual_length: unziped[1]}
        return output_feeds, feed_dict

    def step(self, sess, mode, train_batch):
        output_feeds, feed_dict = self.get_output_feed(mode, train_batch)
        return sess.run(output_feeds, feed_dict=feed_dict)        
