#!/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import random

import tensorflow as tf

import feature_config
from config import CONFIG
import time
import random
import sys

# 增加时效性特征
#train_date=sys.argv[1]

def process_time(features, name, train_date, maxDays=30):
    curTime = features.get(name)
    trainDate = train_date + ' 23:59:59'
    trainTimeArray = time.strptime(trainDate, '%Y%m%d %H:%M:%S')
    trainTime = int(time.mktime(trainTimeArray))
    print('train time : %d'%trainTime)
    if curTime is not None:
        value = tf.floor(tf.divide(trainTime-curTime, 60*60*24))
        value = tf.minimum(value, maxDays)
        value = tf.maximum(value, 0)
    else:
        value = tf.constant(maxDays, dtype=tf.float32)

    value = tf.divide(value, maxDays)
    value = tf.cast(value, dtype=tf.float32)
    #value = tf.Print(value, [value], message='%s value'%name)

    return value

train_date=time.strftime("%Y%m%d", time.localtime())
def process_time2(curTime, maxDays=30):
    trainDate = train_date + ' 23:59:59'
    trainTimeArray = time.strptime(trainDate, '%Y%m%d %H:%M:%S')
    trainTime = int(time.mktime(trainTimeArray))
    print('train time : %d'%trainTime)
    if curTime is not None:
        value = tf.floor(tf.divide(trainTime-curTime, 60*60*24))
        value = tf.minimum(value, maxDays)
        value = tf.maximum(value, 0)
    else:
        value = tf.constant(maxDays, dtype=tf.float32)

    value = tf.divide(value, maxDays)
    value = tf.cast(value, dtype=tf.float32)
    #value = tf.Print(value, [value], message='%s value'%name)
    return value


def process_166(features, name, origin_name, start=0.25, end=0.95, bucket=1001):
    values_196 = features.get(origin_name)
    def bucketed(value):
        step = (end - start)/bucket
        value = (value[0]-start)/step
        value = tf.reshape(value, [1])
        value = tf.cast(value, dtype=tf.int32)
        value = tf.cond(value[0]<0, lambda:tf.constant([0,]), lambda:value)
        value = tf.cond(value[0]>bucket-1, lambda:tf.constant([bucket-1,]), lambda: value)
        #value = tf.Print(value, [value], message='%s bucketed'%origin_name)
        return value

    if values_196 is not None:
        value = bucketed(values_196)
    else:
        value = tf.constant([0])
    value = tf.cast(value, dtype=tf.int64)
    #value = tf.Print(value, [value], message='%s value'%origin_name)
    return value


seq_features=['']

user_features=['']

query_features=['','']


undeepFM=['']

def parse_feature(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features=feature_config.get_feature_format()
    )

    #负样本采样
    label = features['label'][0]


    start_index = 0
    indices_list = []
    values_list = []
    dense_list = []

    # 序列列表
    seq_indices_list=[]
    seq_values_list=[]
    seq_dense_list=[]

    #Attention用户侧列表
    user_indices_list=[]
    user_values_list=[]
    user_dense_list=[]

    #Attention的query侧列表
    query_indices_list=[]
    query_values_list=[]
    query_dense_list=[]

    for config in feature_config.FEATURE_CONFIG:
        if config.dtype == tf.int64:
            not_missing_mask = tf.not_equal(values, -1)
            default_mask = tf.constant([True] + [False]*(config.dims - 1))

            indices = tf.where(
                not_missing_mask,
                values,
                [config.default_value] * config.dims  if not isinstance(config.default_value, list) else config.default_value)

            if config.func:
                indices = config.func(indices, config)

            values = tf.cast(
                tf.logical_or(
                    not_missing_mask,
                    default_mask
                ),
                tf.float32
            )
            # TODO(@qixiang): 需要处理全是0的情况.
            if config.dims != 1:
                values = values / tf.reduce_sum(values, axis=0)
            
            ##如果是deepFM的特征，正常输出
            if config.name not in undeepFM:
                indices_tmp = tf.add(indices, start_index)
                indices_list.append(indices_tmp)
                values_list.append(values)

                start_index += config.size
            ## 输出Attention的Sequence特征
            if config.name in seq_features:
                seq_indices_list.append(indices)
                seq_values_list.append(values)
            ## 输出Attention用户特征
            if config.name in user_features:
                user_indices_list.append(indices)
                user_values_list.append(values)
            ## 输出Attention query特征
            if config.name in query_features:
                query_indices_list.append(indices)
                query_values_list.append(values)

        elif config.dtype == tf.float32:
            if config.name == 'recall':
                values = tf.reshape(values, [6])
                #values = tf.Print(values, [values], message='recall onehot value', summarize=10)
            else:
                values = features[config.name]
            ##如果是deepFM的特征，正常输出 
            if config.name not in undeepFM:
                dense_list.append(values)
            ## 输出Attention的Sequence特征
            if config.name in seq_features:
                seq_dense_list.append(values)
            ## 输出Attention query特征
            if config.name in query_features:
                query_dense_list.append(values)



    if len(dense_list) > 0:
        deepFM = {
            'values': tf.concat(values_list, axis=0),
            'indices': tf.concat(indices_list, axis=0),
            'dense': tf.concat(dense_list, axis=0) if len(dense_list) > 1 else dense_list, 

         }
    else:
        deepFM = {
            'values': tf.concat(values_list, axis=0),
            'indices': tf.concat(indices_list, axis=0),
        }

    if len(seq_dense_list) > 0:
        sequence={
            'values': tf.concat(seq_values_list, axis=0),
            'indices': tf.concat(seq_indices_list, axis=0),
            'dense': tf.concat(seq_dense_list, axis=0) if len(seq_dense_list) > 1 else seq_dense_list[0]
        }
    else:
        sequence={
            'values': tf.concat(seq_values_list, axis=0),
            'indices': tf.concat(seq_indices_list, axis=0),
        }

    if len(user_dense_list) > 0:
        user={
            'values': tf.concat(user_values_list, axis=0),
            'indices': tf.concat(user_indices_list, axis=0),
            'dense': tf.concat(user_dense_list, axis=0) if len(user_dense_list) > 1 else user_dense_list
        }
    else:
        user={
            'values': tf.concat(user_values_list, axis=0),
            'indices': tf.concat(user_indices_list, axis=0),
        }

    if len(query_dense_list) > 0:
        query={
            'values': tf.concat(query_values_list, axis=0),
            'indices': tf.concat(query_indices_list, axis=0),
            'dense': tf.concat(query_dense_list, axis=0) if len(query_dense_list) > 1 else query_dense_list[0]
        }
    else:
        query={
            'values': tf.concat(query_values_list, axis=0),
            'indices': tf.concat(query_indices_list, axis=0),
        }


    features={"deepFM":deepFM,"sequence":sequence,"user":user,"query":query}
 
    #randomSample = tf.random_uniform([],1,10,tf.int64)
    #features = tf.cond(label<1, tf.cond(randomSample<2, lambda:None, lambda:features), lambda:features)
    #label = tf.cond(label<1, tf.cond(randomSample<2, lambda:None, lambda:label), lambda:label)

    return features, label


def input_fn(file_list, is_eval):

    with tf.gfile.Open(file_list) as f:
        file_list = f.read().split()

    print('File list: %d' % len(file_list))

    random.shuffle(file_list)

    files = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = files.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset,
        cycle_length=16, 
        sloppy=True, 
        block_length=64))
    dataset = dataset.map(parse_feature, num_parallel_calls=CONFIG.TRAIN.BATCH_SIZE*4)
    def filter_fun(x, y):
        y = tf.Print(y, [y], message='label value ')
        randomSample = tf.random_uniform([],1,10,tf.int64)
        #randomSample = tf.Print(randomSample, [randomSample], message='random value ')
        label = tf.cond(y<1, lambda:True, lambda:False)
        flag = tf.cond(y<1, lambda:tf.cond(randomSample<3, lambda:False, lambda:True), lambda:True)
        #flag = tf.Print(flag, [flag], message='flag ')
        return flag
    if not is_eval:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
            buffer_size=CONFIG.TRAIN.SHUFFLE_SIZE, count=None))
        dataset = dataset.filter(filter_fun)
    #dataset = dataset.apply(tf.data.experimental.map_and_batch(
    #    map_func=parse_feature, batch_size=CONFIG.TRAIN.BATCH_SIZE, num_parallel_batches=4))
    dataset = dataset.batch(CONFIG.TRAIN.BATCH_SIZE)
    return dataset
