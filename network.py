#!/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import tensorflow as tf

import feature_config
from config import CONFIG
from attention_net import attention_net

ps_num_multipe = 5

def model_analysis():
    import tensorflow.contrib.slim as slim
    model_vars = tf.trainable_variables()
    print('############# Model analysis ##################')
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def build_1st(indices, values, total_size, ps_num):
    with tf.name_scope("1st"):
        weight_table = tf.get_variable(name="1st/weight-bias",
                                       shape=[total_size, 1],
                                       initializer=tf.random_normal_initializer(0.0, 1e-3),
                                       dtype=tf.float32,
                                       partitioner=tf.fixed_size_partitioner(ps_num*ps_num_multipe, axis=0))
        weight = tf.nn.embedding_lookup(params=weight_table, ids=indices, name="weight")
        values = tf.multiply(weight, values)
        logit = tf.reduce_sum(values, axis=1, name="logit")
        if CONFIG.NETWORK.GLOBAL_BIAS:
            bias = tf.get_variable(name="1st/global-bias",
                    shape=[1],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32)
            if CONFIG.NETWORK.EXTRA_SUMMARY:
                tf.summary.scalar("bias/global-bias", bias[0])
            logit = tf.add(logit, bias, name="logit-with-global-bias")
    print("[build_1st] logit:{}".format(logit))
    return logit


def build_2nd(indices, values, total_size, embedding_size, ps_num):
    with tf.name_scope("2nd"):
        embedding_table = tf.get_variable(name="2nd/embedding",
                                          shape=[total_size, embedding_size],
                                          initializer=tf.random_normal_initializer(0.0, 1e-3),
                                          dtype=tf.float32,
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, "embedding"],
                                          partitioner=tf.fixed_size_partitioner(ps_num*ps_num_multipe, axis=0))
        embedding = tf.nn.embedding_lookup(params=embedding_table, ids=indices, name="embedding")
        embedding_values = tf.multiply(embedding, values)
        sum_square_embeddings = tf.square(tf.reduce_sum(embedding_values, axis=1))
        square_sum_embeddings = tf.reduce_sum(tf.square(embedding_values), axis=1)
        second_order_embeddings = 0.5 * tf.subtract(sum_square_embeddings, square_sum_embeddings)

        if CONFIG.NETWORK.FM_EQUAL_SUM:
            logit = tf.reduce_sum(second_order_embeddings, axis=1, name="logit", keepdims=True)
        else:
            logit_weight = tf.get_variable(name="2nd/weight",
                    shape=[embedding_size, 1],
                    initializer=tf.random_normal_initializer(0.0,1e-1),
                    dtype=tf.float32)
            logit = tf.matmul(second_order_embeddings, logit_weight, name="logit")
        print("[build_2nd] embedding:{}\tembedding_values:{}\tsum_square_embeddings:{}\tsquare_sum_embeddings:{}".format(embedding, embedding_values, sum_square_embeddings, square_sum_embeddings))
    print("[build_2nd] logit:{}".format(logit))
    return logit, embedding_values

def build_dnn(dnn_input_values, dense_values,seq_embbeding_value, istraining):
    #print ("dnn_input_values")
    #print (dnn_input_values)

    with tf.name_scope("dnn"):
        if dense_values is not None:
            net = tf.concat([dnn_input_values, dense_values,seq_embbeding_value], axis=-1)
        else:
            net = tf.concat([dnn_input_values,seq_embbeding_value],axis=-1)

        for i, u in enumerate(CONFIG.NETWORK.HIDDEN_UNITS):
            net = tf.layers.dense(
                net,
                u,
                activation=tf.nn.relu,
                kernel_initializer=CONFIG.NETWORK.KERNEL_INITIALIZER,
                kernel_regularizer=CONFIG.NETWORK.REGULARIZER,
                name='layer{}'.format(i)
            )

            if CONFIG.NETWORK.EXTRA_SUMMARY:
                kernel = tf.get_default_graph().get_tensor_by_name("layer{}/kernel:0".format(i))
                tf.summary.histogram("weight/layer{}/kernel".format(i), kernel)
                tf.summary.image("layer{}/kernel".format(i), tf.reshape(kernel, [1, kernel.shape[0], kernel.shape[1], 1]), max_outputs=5)
                bias = tf.get_default_graph().get_tensor_by_name("layer{}/bias:0".format(i))
                tf.summary.histogram("weight/layer{}/bias".format(i), bias)                

            if CONFIG.NETWORK.BATCH_NORM:
                net = tf.layers.batch_normalization(
                    net,
                    training=istraining,
                    name="bn{}".format(i)
                )

            if CONFIG.NETWORK.DEEP_DROPOUT and CONFIG.NETWORK.DEEP_DROPOUT_RATE > 0:
                net = tf.layers.dropout(
                    net,
                    CONFIG.NETWORK.DEEP_DROPOUT_RATE,
                    training=istraining,
                    name="dropout{}".format(i)
                )

        logit = tf.layers.dense(
            net,
            1,
            kernel_initializer=CONFIG.NETWORK.KERNEL_INITIALIZER,
            kernel_regularizer=CONFIG.NETWORK.REGULARIZER,
            name='logit'
        )

    print("[build_dnn] logit:{}".format(logit))
    return logit


def get_field_embedding_values(dims_list, embedding_values):
    embedding_values = tf.unstack(embedding_values, axis=1)

    with tf.name_scope("get_field_embedding_values"):
        field_embedding_list = []
        idx = 0
        for dims in dims_list:
            if dims == 1:
                field_embedding_list.append(embedding_values[idx])
            else:
                field_embedding_list.append(tf.reduce_sum(
                    tf.stack(embedding_values[idx: idx + dims], axis=1),
                    axis=1))
            idx += dims
        field_embedding_values = tf.stack(field_embedding_list, axis=1)
        print("[get_field_embedding_values] embedding_values:{}\tfield_embedding_values:{}".format(embedding_values, field_embedding_values))

    return field_embedding_values


def get_indices_values(features):
    indices = features['indices']
    values = features['values']
    dense_values = features['dense'] if 'dense' in features else None
    return indices,values,dense_values


def build_network(features, istraining, params):

    deepFM_features=features['deepFM']

    indices,values,dense_values = get_indices_values(deepFM_features)
    dims_list, size_list, field_size = feature_config.get_feature_info()
    total_dims, total_size = sum(dims_list), sum(size_list)

    ###embedding_size
    embedding_size = CONFIG.NETWORK.FEATURE_EMBEDDING_SIZE

    print("[build_network] indices:{}\tvalues:{}\tdense_values:{}".format(indices, values, dense_values))
    print("[build_network] dims_list:{} total_dims:{} total_size:{} field_size:{}".format(dims_list, total_dims, total_size, field_size))

    values = tf.reshape(values, [-1, total_dims, 1])

    # 1st
    first_order_logit = build_1st(indices, values, total_size, params['ps_num'])
    

    # 2nd
    second_order_logit, embedding_values = build_2nd(indices, values, total_size, embedding_size, params['ps_num'])

    # attention
    seq_embbeding_value=attention_net(features, params)

    # dnn
    field_embedding_values = get_field_embedding_values(dims_list, embedding_values)
    dnn_input_values = tf.reshape(field_embedding_values, [-1, field_size*embedding_size])
    dnn_logit = build_dnn(dnn_input_values, dense_values,seq_embbeding_value, istraining)

    # debug
    print("[build_network] first_order_logit:{}\tsecond_order_logit:{}\tdnn_logit:{}".format(first_order_logit, second_order_logit, dnn_logit))
    assert not (CONFIG.NETWORK.FM_ONLY and CONFIG.NETWORK.DEEP_ONLY), 'FM_ONLY conflict with DEEP_ONLY'
    
    
    if CONFIG.NETWORK.FM_ONLY:
        logit = tf.add(first_order_logit, second_order_logit)
        if CONFIG.NETWORK.EXTRA_SUMMARY:
            tf.summary.histogram("logit/1st", first_order_logit)
            tf.summary.histogram("logit/2dn", second_order_logit)
            tf.summary.histogram("logit/logit", logit)
    elif CONFIG.NETWORK.DEEP_ONLY:
        logit = dnn_logit
        if CONFIG.NETWORK.EXTRA_SUMMARY:
            tf.summary.histogram("logit/dnn", dnn_logit)
    else:
        logit = tf.add(first_order_logit, second_order_logit)
        logit = tf.add(logit, dnn_logit)
        if CONFIG.NETWORK.EXTRA_SUMMARY:
            tf.summary.histogram("logit/1st", first_order_logit)
            tf.summary.histogram("logit/2dn", second_order_logit)
            tf.summary.histogram("logit/dnn", dnn_logit)
            tf.summary.histogram("logit/logit", logit)

    scores = tf.sigmoid(logit, name='sigmoid')

    scores = tf.squeeze(
        scores,
        axis=1,
        name='scores'
    )

    return scores, logit

def build_model(features, labels, mode, params):
    scores, logit = build_network(features, mode==tf.estimator.ModeKeys.TRAIN, params)

    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'score': scores
        }

        export_outputs = {
            'score': tf.estimator.export.PredictOutput(scores)
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs
        )

    loss = tf.losses.log_loss(labels=labels, predictions=scores, weights=labels*19+1)
    #loss = tf.losses.log_loss(labels=labels, predictions=scores, weights=labels*19+1)
    #loss = tf.losses.log_loss(labels=labels, predictions=scores)


    if CONFIG.NETWORK.REGULARIZER != None:
        reg_loss = tf.losses.get_regularization_loss() 
        loss = tf.add_n([loss, reg_loss], name='reg_loss')

    # Evaluate
    if mode == tf.estimator.ModeKeys.EVAL:
        auc_metric = tf.metrics.auc(labels=labels, predictions=scores, name='auc')
        mae_metric = tf.metrics.mean_squared_error(labels=tf.cast(labels, tf.float32), predictions=scores, name='mae')
        metrics = {
            'auc': auc_metric,
            'mae': mae_metric
        }

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Train
    assert mode == tf.estimator.ModeKeys.TRAIN


    if CONFIG.NETWORK.EXTRA_SUMMARY:
        label_avg_ctr = tf.metrics.mean(values=labels, name="labels_ctr")
        score_avg_ctr = tf.metrics.mean(values=scores, name="scores_ctr")
        tf.summary.scalar("ctr/label", label_avg_ctr)
        tf.summary.scalar("ctr/score", score_avg_ctr)

    tf.summary.histogram('output/labels', labels)
    tf.summary.histogram('output/scores', scores)

    globalStep = tf.train.get_global_step()
    train_op = tf.contrib.layers.optimize_loss(
            loss = loss,
            global_step = globalStep,
            optimizer = CONFIG.NETWORK.OPTIMIZER,
            #learning_rate = CONFIG.NETWORK.LEARNING_RATE,
            learning_rate = tf.train.exponential_decay(CONFIG.NETWORK.LEARNING_RATE, globalStep, 300000, 0.95, staircase=True),
            gradient_multipliers = {
                "2nd/embedding": CONFIG.NETWORK.SPARSE_LEARNING_RATE/CONFIG.NETWORK.LEARNING_RATE if CONFIG.NETWORK.SPARSE_LEARNING_RATE else 1
            },
            clip_gradients= CONFIG.NETWORK.CLIP_GRADIENT,
            learning_rate_decay_fn = lambda lr,step: tf.train.exponential_decay(lr, step, 10000000, 0.96, staircase=True),
            colocate_gradients_with_ops = True,
            summaries = ['global_gradient_norm', 'gradient_norm', 'learning_rate', 'gradients'] if CONFIG.NETWORK.EXTRA_SUMMARY else None)

    model_analysis()

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
