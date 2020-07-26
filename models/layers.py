# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#bias注意力,节点级别注意力
def node_level_attention(adj, H, VS,VK, layer):
    with tf.compat.v1.variable_scope("layer_%s" % layer):

        f1 = tf.matmul(H, VS)
        f2 = tf.matmul(H, VK)

        logits = tf.nn.sigmoid(f1+ tf.transpose(f2, [1, 0]))

        bias_mat=-1e9 * (1.0 - adj)

        coef = tf.nn.softmax(logits + bias_mat)

        return coef,logits



def semantic_level_attention(inputs_list):
    inputs_list=tf.transpose(inputs_list,[1,0,2])

    hiddensize=inputs_list.shape[2]
    VSG = tf.compat.v1.get_variable("VSG" , dtype=tf.float32, shape=(hiddensize, 1))
    VKG = tf.compat.v1.get_variable("VKG" , dtype=tf.float32, shape=(hiddensize, 1))
    f1 = tf.matmul(inputs_list, VSG)
    f2 = tf.matmul(inputs_list, VKG)
    logits=f1+tf.transpose(f2, [0,2, 1])
    logits = tf.nn.sigmoid(logits)

    graphatt = tf.nn.softmax(logits,axis=1, name='graphatt')

    output=tf.matmul(graphatt,inputs_list)
    output=tf.transpose(output,[1,0,2])

    dlist=list(range(output.shape[0]))
    H_list = tf.dynamic_partition(output, dlist, output.shape[0])
    H_list = [tf.reshape(ii, [ii.shape[1], ii.shape[2]]) for ii in H_list]

    return H_list, graphatt



def semantic_level_attention_inductive(inputs_list,hiddensize,nodesize):

    metapath_count=len(inputs_list)
    inputs_list=tf.transpose(inputs_list,[1,0,2])

    VSG = tf.get_variable("VSG" , dtype=tf.float32, shape=(hiddensize, 1))
    VKG = tf.get_variable("VKG" , dtype=tf.float32, shape=(hiddensize, 1))
    f1 = tf.matmul(inputs_list, VSG)
    f2 = tf.matmul(inputs_list, VKG)
    logits=f1+tf.transpose(f2, [0,2, 1])
    logits = tf.nn.sigmoid(logits)

    graphatt = tf.nn.softmax(logits,axis=1, name='graphatt')

    output=tf.matmul(graphatt,inputs_list)
    output=tf.transpose(output,[1,0,2])

    dlist = list(range(metapath_count))
    H_list = tf.dynamic_partition(output, dlist, metapath_count)
    H_list = [tf.reshape(ii, [nodesize, hiddensize]) for ii in H_list]

    return H_list, graphatt


