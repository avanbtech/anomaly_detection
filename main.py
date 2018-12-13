# coding=utf-8
# @author: cer
import os
import tensorflow as tf
from data import *
# from model import Model
from model import Model
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
import pickle
from auto_encoder import AutoEncoder

input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 16
vocab_size = 1871
slot_size = 122
intent_size = 22
epoch_num = 10
epoch_num_ae = 10
model_path = os.getcwd() + "\\store\\model"
use_neg_data = False
train_ae = True

def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                intent_size, epoch_num, None, 
                batch_size, n_layers)
    model.build()
    return model


def train(is_debug=False):
    model = get_model()
    sess = tf.Session()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())

    if (use_neg_data):
        train_data = open("dataset/atis-2.train.w-intent_with_neg.iob", "r").readlines()
    else:
        train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    word2index, index2word, intent2index, index2intent = get_info_from_training_data(train_data_ed)
    index_train = to_index(train_data_ed, word2index, intent2index)
    index_test = to_index(test_data_ed, word2index, intent2index)
    print("%20s%20s%20s" %("Epoch#", "Train Loss", "Intent Accuracy"))
    

    def add_to_vocab_file(fh, data):
        all = set()
        bsize = 16
        for i, batch in enumerate(getBatch(bsize, data)):
            for index in range(len(batch)):
                sen_len = batch[index][1]
                current_vocabs = index_seq2word(batch[index][0], index2word)[:sen_len]
                for w in current_vocabs:
                    if (w in all):
                        continue
                    f_vocab_list.write(w + "\n")
                    all.add(w)

    def add_to_intent_file(fh, data):
        all = set()
        bsize = 16
        for i, batch in enumerate(getBatch(bsize, data)):
            for index in range(len(batch)):
                sen_len = batch[index][1]
                w = index2intent[batch[index][2]]
                if (w in all):
                    continue
                f_vocab_list.write(w + "\n")
                all.add(w)
    
    f_vocab_list = open("vocab_list.in", "w")
    add_to_vocab_file(f_vocab_list, index_train)
    add_to_vocab_file(f_vocab_list, index_test)
    f_vocab_list.close()

    f_vocab_list = open("intent_list.in", "w")
    add_to_intent_file(f_vocab_list, index_train)
    add_to_intent_file(f_vocab_list, index_test)
    f_vocab_list.close()
    
    # saver = tf.train.Saver()

    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(getBatch(batch_size, index_train)):
            _, loss, intent, _ = model.step(sess, "train", batch)            
            train_loss += loss            
        train_loss /= (i + 1)        
        
        intent_accs = []
        for j, batch in enumerate(getBatch(batch_size, index_test)):
            intent, _ = model.step(sess, "test", batch)
            intent_acc = accuracy_score(list(zip(*batch))[2], intent)
            intent_accs.append(intent_acc)
        print("%20d%20f%20f" %(epoch, train_loss, np.average(intent_accs)))
        
    print("Training auto-encoder...")
    print("%20s%20s%20s%20s%20s" %("Epoch#", "Train Loss", "Neg Data Loss", "Good Data Loss", "Ratio"))
    ae_model = AutoEncoder(model)
    ae_model.tf_init(sess)
    
    if (train_ae):
        for epoch in range(epoch_num_ae):
            mean_loss = 0.0
            train_loss = 0.0
            for i, batch in enumerate(getBatch(batch_size, index_train)):
                intent, _, _, output_true, output_layer, loss, _  = ae_model.step(sess, "train", batch)            
                train_loss += loss            
            train_loss /= (i + 1)              
            result1, result2 = run_batch_test(ae_model, sess, word2index, index2intent, index_test, epoch)
            r = (result1 - result2) / result1 * 100
            print("%20d%20f%20f%20f%20f" %(epoch, train_loss, result1, result2, r))
    else:
        run_batch_test(ae_model, sess, word2index, index2intent, index_test, 0)
    # run_test(ae_model, sess, word2index, index2intent)

def get_ids(str, word2index):
    vocabs = str.split(' ')
    ids = []
    keys = word2index.keys()
    for v in vocabs:
        if (v in keys):
            ids.append(word2index[v])
        else:
            ids.append(word2index["<UNK>"])
    i_actual = len(ids)
    ids.append(word2index["<EOS>"])
    for i in range(len(ids), 50):
        ids.append(word2index["<PAD>"])
    batch = []
    for i in range(batch_size):
        batch.append([ids, i_actual])
    return batch

def run_batch_test(model, sess, word2index, index2intent, index_test, epoch):
    test_data = open("dataset/uns_test.txt", "r").readlines()
    total_cnt = 0
    error = 0    
    total_loss = 0.0
    neg_data_loss = []
    for str in test_data:
        batch = get_ids(str, word2index)
        intent, intent_logits, output_true, output_layer, loss, loss_per_entry  = model.step(sess, "test", batch)
        total_cnt += 1
        if (train_ae):
            total_loss += loss_per_entry[0]
            neg_data_loss.append(loss_per_entry[0])
        else:
            if (use_neg_data):
                is_error = index2intent[intent[0]] != 'atis_unsupported'
            else:
                logits = intent_logits[0]
                logits.sort()
                l1 = logits[-1]
                l2 = logits[-2]
                diff = (l1 - l2) / np.max([np.abs(l1), np.abs(l2)]) * 100
                is_error = diff < 80
            if (is_error):
                # print('error sentence: %s' %(str))
                error += 1
    total_loss /= total_cnt    
    if (train_ae):
        result1 = total_loss
        # print('Negative data results: Average loss: %f' %(total_loss))
    else:
        result1 = 100 * (total_cnt - error) / total_cnt
        print('Negative data results: Total: %d, error: %d, accuracy: %f' %(total_cnt, error, result1))
    total_cnt = 0
    error = 0
    total_loss = 0.0
    good_data_loss = []
    for i, batch in enumerate(getBatch(batch_size, index_test)):
        intent, intent_logits, output_true, output_layer, loss, loss_per_entry  = model.step(sess, "test", batch)
        count = 0
        for cur_int in intent:
            if (train_ae):
                total_loss += loss_per_entry[count]
                good_data_loss.append(loss_per_entry[count])
            else:
                if (use_neg_data):
                    is_error = index2intent[cur_int] == 'atis_unsupported'
                else:
                    logits = intent_logits[count]
                    logits.sort()
                    l1 = logits[-1]
                    l2 = logits[-2]
                    diff = (l1 - l2) / np.max([np.abs(l1), np.abs(l2)]) * 100
                    is_error = diff < 80                
                if (is_error):                
                    error += 1
            total_cnt += 1
            count += 1
    total_loss /= total_cnt
    if (train_ae):
        result2 = total_loss
        # print('Good data results: Average loss: %f' %(total_loss))
        fname = "results/ae_neg_loss_%d.txt" %(epoch)
        fh = open(fname, "w")
        for aLoss in neg_data_loss:
            fh.write("%f\n" %(aLoss))
        fh.close()
        fname = "results/ae_good_loss_%d.txt" %(epoch)
        fh = open(fname, "w")
        for aLoss in good_data_loss:
            fh.write("%f\n" %(aLoss))
        fh.close()
    else:
        result2 = 100 * (total_cnt - error) / total_cnt
        print('Good data results: Total: %d, error: %d, accuracy: %f' %(total_cnt, error, result2))

    return result1, result2

def run_test(model, sess, word2index, index2intent):
    while True:
        str = input('Please enter your utterance: ')
        batch = get_ids(str, word2index)
        intent, _, output_true, output_layer, loss  = model.step(sess, "test", batch)
        print('Intent: %s, loss: %f' %(index2intent[intent[0]], loss))


def test():
    global batch_size
    batch_size = 1
    model = get_model()
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_path + '-9765')

    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)

    run_test(model, sess, word2index, index2intent)

if __name__ == '__main__':
    to_train = True
    if (to_train):
        train()
    else:
        test()
