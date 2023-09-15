import csv
import datetime
import os.path
import time

import numpy as np
import pandas as pd
import tensorflow as tf


class WideAndDeepModel:
    def __init__(self, wide_length, deep_length, deep_last_layer_len, softmax_label):
        self.input_wide_part = tf.placeholder(tf.float32, shape=[None, wide_length], name='input_wide_part')
        self.input_deep_part = tf.placeholder(tf.float32, shape=[None, deep_length], name='input_deep_part')
        self.input_y = tf.placeholder(tf.float32, shape=[None, softmax_label], name='input_y')

        with tf.name_scope('deep_part'):
            w_x1 = tf.Variable(tf.random_normal([wide_length, 256], stddev=0.03), name='w_x1')
            b_x1 = tf.Variable(tf.random_normal([256]), name='b_x1')
            w_x2 = tf.Variable(tf.random_normal([256, deep_last_layer_len], stddev=0.03), name='w_x2')
            b_x2 = tf.Variable(tf.random_normal([deep_last_layer_len]), name='b_x2')

            z1 = tf.add(tf.matmul(self.input_wide_part, w_x1), b_x1)
            a1 = tf.nn.relu(z1)
            self.deep_logits = tf.add(tf.matmul(a1, w_x2), b_x2)

        with tf.name_scope('wide_part'):
            weights = tf.Variable(tf.truncated_normal([deep_last_layer_len + wide_length, softmax_label]))
            biases = tf.Variable(tf.zeros(softmax_label))

            self.wide_and_deep = tf.concat([self.deep_logits, self.input_wide_part], axis=1)
            self.wide_and_deep_logits = tf.add(tf.matmul(self.wide_and_deep, weights), biases)
            self.predictions = tf.argmax(self.wide_and_deep_logits, 1, name='prediction')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.wide_and_deep_logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.predictions, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    def load_data_and_labels(self, path):
        data = []
        y = []
        total_q = []

        with open(path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                emb_val = row[4].split(';')
                emb_val_f = [float(i) for i in emb_val]

                cate_emp = row[5].split(';')
                cate_emp_val_f = [float(i) for i in cate_emp]

                total_q.append(int(row[3]))
                data.append(emb_val_f + cate_emp_val_f)
                y.append(float(row[1]))
        data = np.asarray(data)
        total_q = np.asarray(total_q)
        y = np.asarray(y)

        bins = pd.qcut(y, 50, retbins=True)

        def convert_label_to_interval(self, y):
            gmv_bins = []
            for i in range(len(y)):
                interval = int(y[i] / 20000)
                if interval < 1000:
                    gmv_bins.append(interval)
                elif interval >= 1000:
                    gmv_bins.append(1000)

            gmv_bins = np.asarray(gmv_bins)
            return gmv_bins

        y = convert_label_to_interval(y)

        def dense_to_one_hot(self, labels_dense, num_classes):
            num_labels = labels_dense.shape[0]
            index_offset = np.arage(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
            return labels_one_hot

        labels_count = 1001
        labels = dense_to_one_hot(y, labels_count)
        labels = labels.astype(np.uint8)

        def dense_to_one_hot2(labels_dense, num_classes):
            num_labels = labels_dense.shape[0]
            index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[index_offset + labels_dense.ravel() - 1] = 1

            return labels_one_hot

        total_q_classes = np.unique(total_q).shape[0]
        total_q = dense_to_one_hot2(total_q, total_q_classes)

        data = np.concatenate((data, total_q), axis=1)

        return data, labels

    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    if __name__ == '__main__':
        load_data_and_labels('train.csv')

