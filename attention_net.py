#!/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import tensorflow as tf

import feature_config
from config import CONFIG

ps_num_multipe = 5

def build_embedding(name,indices,values,size,embedding_size,ps_num):
    with tf.name_scope("attention"):
        embedding_table = tf.get_variable(name="attention/{}".format(name),
                                        shape=[size, embedding_size],
                                        initializer=tf.random_normal_initializer(0.0, 1e-3),
                                        dtype=tf.float32,
                                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, name],
                                        partitioner=tf.fixed_size_partitioner(ps_num*ps_num_multipe, axis=0))
        embedding = tf.nn.embedding_lookup(params=embedding_table, ids=indices, name=name)
	
        embedding_value = tf.multiply(embedding, values)
    return embedding_value 

def build_seq_embedding(name,seq_dense_values,indices,values,size_list,embedding_size,ps_num):
    seq_indices = tf.unstack(indices, axis=1) ##低阶tensor组成的list
    seq_values = tf.unstack(values, axis=1)
    ## 序列 embedding
    embedding_values =[]
    vector_values = tf.split(seq_dense_values,10,1)    ##保存每个用户最近10个行为序列的16维dssm 
    for i in range(10):   ##每个用户保存最近10个行为序列
        ## uuid
        uuid_indice = seq_indices[i]
        uuid_value = seq_values[i]
        uuid_size = size_list[0]
        ## hour
        hour_indice = seq_indices[i+10]
        hour_value = seq_values[i+10]
        hour_size = size_list[1]

        ## day
        day_indice = seq_indices[i+20]
        day_value = seq_values[i+20]
        day_size = size_list[2]

        ## uuid embedding
        uuid_embedding_value = build_embedding("seq_uuid_embedding_{}".format(str(i)),uuid_indice,uuid_value,uuid_size,embedding_size,ps_num)
        ## hour embedding
        hour_embedding_value = build_embedding("seq_hour_embedding_{}".format(str(i)),hour_indice,hour_value,hour_size,embedding_size,ps_num)
        ## day embedding
        #day_embedding_value = build_embedding("seq_day_embedding_{}".format(str(i)),day_indice,day_value,day_size,embedding_size,ps_num)
        vector_embedding_value = vector_values[i]
        embedding_value = tf.concat([uuid_embedding_value,hour_embedding_value,vector_embedding_value],axis=-1)

        #embedding_value = uuid_embedding_value
        embedding_values.append(embedding_value)       
    return embedding_values 



def build_user_embedding(name,indices,values,size_list,embedding_size,ps_num):
    ###对每一类特征单独Embedding，所以要将tensor分解成低阶list，然后取每个特征进行Embedding 
    ### 目前只有did一个特征
    indices = tf.unstack(indices, axis=1) ##低阶tensor组成的list
    values = tf.unstack(values, axis=1)
    did_indice =indices[0]
    did_value = values[0] 
    did_size = size_list[0]
    user_embedding = build_embedding(name,did_indice,did_value,did_size,embedding_size,ps_num)
    return user_embedding


def build_query_embedding(name,indices,values,size_list,embedding_size,ps_num):
    ###对每一类特征单独Embedding，所以要将tensor分解成低阶list，然后取每个特征进行Embedding 
    indices = tf.unstack(indices, axis=1) ##低阶tensor组成的list
    values = tf.unstack(values, axis=1)
    uuid_indice = indices[0]
    uuid_value = values[0]
    uuid_size = size_list[0]
    query_embedding = build_embedding(name,uuid_indice,uuid_value,uuid_size,embedding_size,ps_num)
    return query_embedding


###attention net
def build_attention_net(user_embedding,query_embedding,seq_embeddings):
    attention_params=[16,8,1] 
    a = []
    for j in range(10):
        with tf.name_scope("attention/{}".format(j)):
            att_net = tf.concat([user_embedding,query_embedding,seq_embeddings[j]], axis=-1)
      
            for i, u in enumerate(attention_params):
                att_net = tf.layers.dense(
                    att_net,
                    u,
                    activation=tf.nn.relu,
                    kernel_initializer=CONFIG.NETWORK.KERNEL_INITIALIZER,
                    kernel_regularizer=CONFIG.NETWORK.REGULARIZER,
                    name='layer{}_{}'.format(j,i)
                )
        a.append(att_net)
    a = tf.stack(a,axis=1)
    a = tf.nn.softmax(a) 
    seq_embedding = tf.stack(seq_embeddings,axis=1)

    seq_output = tf.reduce_sum(tf.multiply(seq_embedding,a),axis=1)
    return seq_output
            


def get_indices_values(features):
    indices = features['indices']
    values = features['values']
    dense_values = features['dense'] if 'dense' in features else None
    return indices,values,dense_values

def attention_net(features, params):

    sequence_features=features['sequence']
    user_features = features['user']
    query_features = features['query']

    ###embedding_size
    embedding_size = CONFIG.NETWORK.FEATURE_EMBEDDING_SIZE

    ##user embedding
    user_indices,user_values,user_dense_values = get_indices_values(user_features)
    user_dims_list, user_size_list, user_field_size = feature_config.get_user_info()
    user_total_dims, user_total_size = sum(user_dims_list), sum(user_size_list)
    user_values = tf.reshape(user_values,[-1,user_total_dims,1])
    user_embedding = build_user_embedding("user_embedding",user_indices,user_values,user_size_list,embedding_size,params['ps_num'])
    
    ##query embedding
    query_indices,query_values,query_dense_values = get_indices_values(query_features)
    query_dims_list, query_size_list, query_field_size = feature_config.get_query_info()
    query_total_dims, query_total_size = sum(query_dims_list), sum(query_size_list)
    query_values = tf.reshape(query_values,[-1,query_total_dims,1])
    query_embedding = build_query_embedding("query_embedding",query_indices,query_values,query_size_list,embedding_size,params['ps_num'])
    
    ##seq embedding
    seq_indices,seq_values,seq_dense_values = get_indices_values(sequence_features)
    seq_dims_list, seq_size_list, seq_field_size = feature_config.get_sequence_info()
    seq_total_dims,seq_total_size = sum(seq_dims_list), sum(seq_size_list)
    seq_values = tf.reshape(seq_values,[-1,seq_total_dims,1])
    seq_embedding = build_seq_embedding("seq_embedding",seq_dense_values,seq_indices,seq_values,seq_size_list,embedding_size,params['ps_num'])


    # attention
    #seq_dense_values = tf.expand_dims(seq_dense_values, -1)
    seq_embbeding_value=build_attention_net(user_embedding,query_dense_values,seq_embedding)

    return seq_embbeding_value 

