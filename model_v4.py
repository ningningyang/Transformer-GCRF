#coding=utf-8
import tensorflow as tf
import numpy as np
import pdb
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops import gen_array_ops
from tensorflow.losses import Reduction
from loss_sc import *
from decode_sc import *
from ntransformer import *
import copy

def init_position_embedding(maxlen, E):
    position_enc = np.array([
        [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
        for pos in range(maxlen)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc

class model_ig():
    def __init__(self, batch_size, embeddings, embedding_size, update_embedding, hidden_dim, num_tags, bertconfig, max_length, l2_lambda=None):
        self.batch_size = batch_size
        self.memory_length = 8
        self.embedding_size = embedding_size
        self.embeddings = embeddings
        self.update_embedding = update_embedding
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.l2_lambda = l2_lambda
        self.max_length = max_length
        self.bertconfig = bertconfig
        x_state = self.encode_x_sentence()
        # x_state [batch_size, seq_len_x, hidden_dim]
        self.logits = self.mlp_predict(x_state)
        # logits [batch_size, seq_len, num_tags]
        # print("self.logits:", self.logits)
        log_likelihood, self.transition_params1, self.transition_params2 \
                                               = crf_log_likelihood(self.logits, 
                                                                    self.input_yg,
                                                                    self.sequence_lengths_reshape,
                                                                    self.inputs_one_row_index,
                                                                    self.inputs_reshape_index,
                                                                    self.inputs_reshape_shape,
                                                                    self.tag_indices_reshape_shape,
                                                                    self.one_column_head_index,
                                                                    self.one_column_head_matrix_index,
                                                                    self.one_column_pro,
                                                                    self.one_column_pro_matrix_index,
                                                                    self.one_column_matrix_shape,
                                                                    self.not_one_column_index,
                                                                    self.has_known_tag,
                                                                    self.has_skip_word,
                                                                    self.has_head_index,
                                                                    transition_params=None
                                                                    )

        self.prediction = crf_decode(self.logits, 
                                    self.transition_params1, 
                                    self.transition_params2, 
                                    self.sequence_lengths_reshape,
                                    self.inputs_one_row_index,
                                    self.inputs_reshape_index,
                                    self.inputs_reshape_shape,
                                    self.tag_indices_reshape_shape,
                                    self.one_column_head_index,
                                    self.one_column_head_matrix_index,
                                    self.one_column_pro,
                                    self.one_column_pro_matrix_index,
                                    self.one_column_matrix_shape,
                                    self.not_one_column_index,  
                                    self.has_known_tag,
                                    self.has_skip_word,
                                    self.has_head_index,
                                    self.batch_size_reshape,
                                    )
        # vars = tf.trainable_variables()
        # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])
        # self.loss = -log_likelihood + self.l2_lambda * lossL2
        self.loss = -log_likelihood
        # with tf.name_scope("loss"):
        #     cross_entropy_g = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_yg, name='loss_g')
        #     mask = tf.sequence_mask(self.sequence_lengths)
        #     self.loss_g = tf.boolean_mask(cross_entropy_g, mask)
        #     self.loss = tf.reduce_mean(self.loss_g)
        # self.prediction = tf.argmax(self.logits, axis=2)
        
    def get_word_embedding(self, input_data):
        input_x_ = tf.expand_dims(input_data, axis=[-1])
        flat_input_ids = tf.reshape(input_x_, [-1])
        output = tf.gather(self.word_embedding, flat_input_ids)
        input_shape = get_shape_list(input_x_)
        output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        output = tf.nn.dropout(output, self.dropout_keep_prob)
        output = dense_layer_2d(
            output, self.hidden_dim, create_initializer(self.bertconfig.initializer_range),
            None, name="embedding_hidden_mapping_in")
        output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * self.hidden_dim])
        return output, input_shape

    def get_part_embedding(self, input_part, input_shape):
        flat_token_type_ids = tf.reshape(input_part, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.bertconfig.type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_embedding)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           input_shape[0:-1] + [input_shape[-1] * self.hidden_dim])
        return token_type_embeddings

    def get_position_embedding(self, input_sum_data, input_shape):
        position_embeddings = tf.slice(self.full_position_embedding, [0, 0], [input_shape[1], -1])
        num_dims = len(input_sum_data.shape.as_list())
        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        position_broadcast_shape.extend([input_shape[1], self.hidden_dim])
        position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
        return position_embeddings

    def encode_x_sentence(self):
        # '''
        # encode 输入的x
        # :return: x_state [batch_size, seq_len_x, hidden_dim*2]
        # '''
        with tf.name_scope('placeholders'):
            self.input_yg = tf.placeholder(tf.int32, [self.batch_size, None], name="input_yg")
            self.input_yi = tf.placeholder(tf.int32, [self.batch_size, None], name="input_yi")
            self.sequence_lengths = tf.placeholder(tf.int32, [self.batch_size], name="sequence_lengths")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.memory_data = tf.placeholder(tf.int32, [self.batch_size, None], name="memory_data")
            self.memory_mask = tf.placeholder(tf.int32, [self.batch_size, None, None], name="memory_mask")
            self.memory_mask2 = tf.placeholder(tf.int32, [self.batch_size, None], name="memory_mask2")
            self.memory_tag = tf.placeholder(tf.int32, [self.batch_size, None], name="memory_tag")
            self.memory_pos = tf.placeholder(tf.int32, [self.batch_size, None], name="memory_pos")
            self.sen_data = tf.placeholder(tf.int32, [self.batch_size, None], name="sen_data")
            self.sen_mask = tf.placeholder(tf.int32, [self.batch_size, None], name="sen_mask")
            self.sen_tag = tf.placeholder(tf.int32, [self.batch_size, None], name="sen_tag")
            self.hidden_dropout_prob = tf.placeholder(tf.float32, name="hidden_dropout_prob")
            self.attention_probs_dropout_prob = tf.placeholder(tf.float32, name="attention_probs_dropout_prob")
            self.sequence_lengths_reshape = tf.placeholder(tf.int32, [None], name="sequence_lengths_reshape")
            self.inputs_one_row_index = tf.placeholder(tf.int32, [None, 2], name="inputs_one_row_index")
            self.inputs_reshape_index = tf.placeholder(tf.int32, [None, 2], name="inputs_reshape_index")
            self.inputs_reshape_shape = tf.placeholder(tf.int32, [3], name="inputs_reshape_shape")
            self.tag_indices_reshape_shape = tf.placeholder(tf.int32, [2], name="tag_indices_reshape_shape")
            self.one_column_head_index = tf.placeholder(tf.int32, [None, 1], name="one_column_head_index")
            self.one_column_head_matrix_index = tf.placeholder(tf.int32, [None, 1], name="one_column_head_matrix_index")
            self.one_column_pro = tf.placeholder(tf.int32, [None], name="one_column_pro")
            self.one_column_pro_matrix_index = tf.placeholder(tf.int32, [None, 1], name="one_column_pro_matrix_index")
            self.one_column_matrix_shape = tf.placeholder(tf.int32, [2], name="one_column_matrix_shape")
            self.not_one_column_index = tf.placeholder(tf.int32, [None, 1], name="not_one_column_index")
            self.has_known_tag = tf.placeholder(tf.bool, name="has_known_tag")
            self.has_skip_word = tf.placeholder(tf.bool, name="has_skip_word")
            self.has_head_index = tf.placeholder(tf.bool, name="has_head_index")
            self.batch_size_reshape = tf.placeholder(tf.int32, name="batch_size_reshape")

        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            # self.word_embedding = tf.get_variable(
            #                                  name="word_embedding",
            #                                  shape=[self.bertconfig.vocab_size, self.hidden_dim],
            #                                  initializer=create_initializer(self.bertconfig.initializer_range))
            self.token_type_embedding = tf.get_variable(
                                             name="token_type_embedding",
                                             shape=[self.bertconfig.type_vocab_size, self.hidden_dim],
                                             initializer=create_initializer(self.bertconfig.initializer_range))
            # self.token_type_embedding_2 = tf.get_variable(
            #                                 name="token_type_embedding_2",
            #                                 shape=[self.memory_length, self.hidden_dim],
            #                                 initializer=create_initializer(self.bertconfig.initializer_range))
            # self.full_position_embedding = tf.get_variable(
            #                                  name="full_position_embedding",
            #                                  shape=[self.bertconfig.max_position_embeddings, self.hidden_dim],
            #                                  initializer=create_initializer(self.bertconfig.initializer_range))
            self.word_embedding = tf.Variable(self.embeddings, 
                                                   dtype=tf.float32, trainable=True,
                                                   name="word_embedding")
            self.full_position_embedding = tf.Variable(init_position_embedding(self.bertconfig.max_position_embeddings, self.hidden_dim), 
                                                   dtype=tf.float32, trainable=False,
                                                   name="full_position_embedding")

            self.sen_x_word_embedding, input_shape = self.get_word_embedding(self.sen_data)
            # self.sen_x_word_embedding = self.sen_x_word_embedding * tf.sqrt(self.hidden_dim*1.)
            # word embedding [batch, seq_len, embedding_size]
            token_type_embeddings = self.get_part_embedding(self.sen_tag, input_shape)
            self.sen_x_word_embedding += token_type_embeddings
            position_embeddings = self.get_position_embedding(self.sen_x_word_embedding, input_shape)
            self.sen_x_word_embedding += position_embeddings
            self.sen_output = layer_norm_and_dropout(self.sen_x_word_embedding, self.hidden_dropout_prob)
            self.memory_word_embedding, memory_shape = self.get_word_embedding(self.memory_data)
            # self.memory_word_embedding = self.memory_word_embedding * tf.sqrt(self.hidden_dim*1.)
            mem_token_type_embeddings = self.get_part_embedding(self.memory_tag, memory_shape)
            self.memory_word_embedding += mem_token_type_embeddings
            # mem_position_embeddings = self.get_position_embedding(self.memory_word_embedding, memory_shape)
            flat_token_type_ids = tf.reshape(self.memory_pos, [-1])
            one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.bertconfig.max_position_embeddings)
            position_embeddings = tf.matmul(one_hot_ids, self.full_position_embedding)
            mem_position_embeddings = tf.reshape(position_embeddings, memory_shape[0:-1] + [memory_shape[-1] * self.hidden_dim])                
            self.memory_word_embedding += mem_position_embeddings
            self.memoty_output = layer_norm_and_dropout(self.memory_word_embedding, self.hidden_dropout_prob)

        with tf.name_scope('encoder'):
            self.all_encoder_layers = encoder_model(
                input_tensor=self.memoty_output,
                attention_mask=self.memory_mask,
                hidden_size=self.bertconfig.hidden_size,
                num_hidden_layers=self.bertconfig.num_hidden_layers,
                num_attention_heads=self.bertconfig.num_attention_heads,
                intermediate_size=self.bertconfig.intermediate_size,
                intermediate_act_fn=get_activation(self.bertconfig.hidden_act),
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                initializer_range=self.bertconfig.initializer_range,
                do_return_all_layers=True)
            self.memoty_encode_output = self.all_encoder_layers[-1]

        with tf.name_scope('decoder'):
            attention_mask1 = create_attention_mask_from_input_mask(self.sen_data, self.sen_mask)
            attention_mask2 = create_attention_mask_from_input_mask(self.sen_data, self.memory_mask2)
            # print("self.sen_output:", self.sen_output)
            # print("self.memoty_encode_output:", self.memoty_encode_output)
            self.all_encoder_layers, self.last_layer_att = decoder_model(
                input_tensor=self.sen_output,
                input_tensor2=self.memoty_encode_output,
                attention_mask=attention_mask1,
                attention_mask2=attention_mask2,
                hidden_size=self.bertconfig.hidden_size,
                num_hidden_layers=self.bertconfig.num_hidden_layers,
                num_attention_heads=self.bertconfig.num_attention_heads,
                intermediate_size=self.bertconfig.intermediate_size,
                intermediate_act_fn=get_activation(self.bertconfig.hidden_act),
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                initializer_range=self.bertconfig.initializer_range,
                do_return_all_layers=True)
            self.final_output = self.all_encoder_layers[-1]        

        return self.final_output

    def mlp_predict(self, x_state):
        #x_state = tf.reshape(tf.concat([x_state, sentence_x_state, word_x_state], axis=-1), [-1, self.hidden_dim*6])
        self.shape = tf.shape(x_state)
        x_state = tf.reshape(x_state, [-1, self.hidden_dim])
        W3 = tf.get_variable(name="W3",
                             shape=[self.hidden_dim, self.hidden_dim/2],#shape=[6*self.hidden_dim, self.hidden_dim],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             dtype=tf.float32)
        b3 = tf.get_variable(name="b3",
                             shape=[self.hidden_dim/2],
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32)
        temp_state = tf.nn.tanh(tf.matmul(x_state, W3) + b3)
        temp_state = tf.nn.dropout(temp_state, self.dropout_keep_prob)
        W4 = tf.get_variable(name="W4",
                             shape=[self.hidden_dim/2, self.num_tags],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             dtype=tf.float32)
        b4 = tf.get_variable(name="b4",
                             shape=[self.num_tags],
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32)
        logits = tf.reshape(tf.matmul(temp_state, W4) + b4, [self.batch_size, self.shape[1], self.num_tags])
        # logits = tf.nn.dropout(logits_temp, self.dropout_keep_prob)
        # logits [batch_size, seq_len_x, num_tags]
        #self.prediction = tf.argmax(logits, axis=2)
        # prediction [batch_size, seq_len_x]
        return logits
