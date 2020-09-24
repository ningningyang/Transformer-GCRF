#coding=utf-8
import tensorflow as tf
from data_util import *
from model_v4 import model_ig
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import datetime
import numpy as np
import pandas as pd
import random
import json
import pdb
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ntransformer import *
import sys

np.set_printoptions(threshold=sys.maxsize)

def get_weighted_f1(f1_dict):
    ans = 0.
    weight_list = [0.,0.3227,0.2119,0.0694,0.1279,0.0576,0.0287,0.0261,0.0422,0.0351,0.0329,0.0093,0.0118,0.0006,0.0067,0.0163,0.0008]
    for label in range(17):
        if str(label) in f1_dict.keys():
            ans += f1_dict[str(label)]['f1-score'] * weight_list[label]
    return ans

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)/20.

#np.set_printoptions(threshold='nan')
flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.005, "Learning rate for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm", 10.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 8, "Batch size for training.")
flags.DEFINE_integer("epochs", 60, "Number of epochs to train for.")
flags.DEFINE_integer("embedding_size", 300, "Embedding size for embedding matrices.")
flags.DEFINE_integer("hidden_dim", 512, "Dimension of hidden state in lstm.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_integer("evaluate_every", 500, "Evaluate and print results every x epochs")
flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
flags.DEFINE_integer("pretrain_embedding", 1, "Load pretrain embedding or not.")
flags.DEFINE_integer("vocab_size", 17198 + 1, "Load pretrain embedding or not.")
flags.DEFINE_boolean("update_embedding", False, "Train embedding or not.")
flags.DEFINE_integer("num_tags", 17, "2-classification or 17-classification.")
flags.DEFINE_float("l2_lambda", 1e-5, "Regularization coefficient.")
flags.DEFINE_float("max_length", 134,"Max sequence length for transformer.")

FLAGS = flags.FLAGS

def str1_2_list(row):
    row_list = [list(map(eval, row[i].lstrip('[').rstrip(']').split(','))) for i in range(len(row))]
    return row_list

def str2_2_list(row):
    rows_list = []
    for i in range(len(row)):
        row_list = row[i].split('], [')
        each_row = [list(map(eval, row_list[i].lstrip('[').rstrip(']').split(','))) for i in range(len(row_list))]
        rows_list.append(each_row)
    return rows_list

# 载入数据并区分训练、验证、测试集
# data = pd.read_csv('./sms_new_5_2_fill.csv', encoding="utf_8_sig")
data = pd.read_csv('./seq_data_tjz_fill_p_5_2.csv', encoding="utf_8_sig")

xml_number = data.xml_number
utt_index = str1_2_list(data.utt_index)
label_g = str1_2_list(data.label_g)
label_i = str1_2_list(data.label_i)
memory_g = str2_2_list(data.memory)
x_part = str1_2_list(data.x_part)
memory_g_p = str2_2_list(data.memory_part)

train_data = []
dev_data = []
test_data = []
#td_data = []

train_labelg = []
dev_labelg = []
test_labelg = []
#td_labelg = []

train_labeli = []
dev_labeli = []
test_labeli = []
#td_labeli = []

train_memoryg = []
dev_memoryg = []
test_memoryg = []
#td_memoryg = []

train_data_part = []
dev_data_part = []
test_data_part = []
#td_data_part = []

train_sc = []
dev_sc = []
test_sc = []
#td_sc = []

train_memoryg_part = []
dev_memoryg_part = []
test_memoryg_part = []
#td_memoryg_part = []

loss_train = []
loss_dev = []
for j, num in enumerate(xml_number):
    if num>=99 and num < 586:
        train_data.append(utt_index[j])
        train_labelg.append(label_g[j])
        train_labeli.append(label_i[j])
        train_memoryg.append(memory_g[j])
        train_data_part.append(x_part[j])
        train_memoryg_part.append(memory_g_p[j])
    elif num >= 586:
        dev_data.append(utt_index[j])
        dev_labelg.append(label_g[j])
        dev_labeli.append(label_i[j])
        dev_memoryg.append(memory_g[j])
        dev_data_part.append(x_part[j])
        dev_memoryg_part.append(memory_g_p[j])
    else:
        test_data.append(utt_index[j])
        test_labelg.append(label_g[j])
        test_labeli.append(label_i[j])
        test_memoryg.append(memory_g[j])
        test_data_part.append(x_part[j])
        test_memoryg_part.append(memory_g_p[j])

#print('data load finished')
print('train_size: ', len(train_data))
print('dev_size: ', len(dev_data))
#print('train_dev_size:', len(td_data))
print('test_size: ', len(test_data))

#with open("vocab_sms_new_fill.json", 'r') as f:
with open("vocab_fill_p.json", 'r') as f:
    str1 = f.read()
    my_dict = json.loads(str1)
dict_word = []
dict_index = []
for a, b in my_dict.items():
    dict_word.append(a)
    dict_index.append(b)

# if FLAGS.pretrain_embedding == 1:
# embeddings = random_embedding(FLAGS.vocab_size, FLAGS.embedding_size)
# else:
# embedding_path = 'new_np.npy'
embedding_path = 'np.npy'
embeddings = np.array(np.load(embedding_path), dtype='float32')

bertconfig = BertConfig(
          vocab_size=FLAGS.vocab_size,
          hidden_size=FLAGS.hidden_dim,
          num_hidden_layers=4,
          num_attention_heads=4,
          intermediate_size=4*FLAGS.hidden_dim,
          hidden_act="gelu",
          hidden_dropout_prob=0.1,
          attention_probs_dropout_prob=0.1,
          max_position_embeddings=128,
          type_vocab_size=3,
          initializer_range=0.02)

with tf.Session() as sess:
    model_ig = model_ig(FLAGS.batch_size, embeddings, FLAGS.embedding_size, FLAGS.update_embedding, FLAGS.hidden_dim,
                  FLAGS.num_tags, bertconfig, FLAGS.max_length, FLAGS.l2_lambda)

    # Define Training procedure
    global_step1 = tf.Variable(0, name="global_step1", trainable=False)
    global_step2 = tf.Variable(0, name="global_step2", trainable=False)
    init_lr = tf.math.rsqrt(tf.cast(FLAGS.hidden_dim, tf.float32))
    # lr1 = noam_scheme(init_lr, global_step1, warmup_steps=8000.)
    # lr2 = noam_scheme(init_lr * 2., global_step2, warmup_steps=8000.) 
    lr1 = 3e-5
    lr2 = 3e-5 * 2.
    optimizer1 = tf.train.AdamOptimizer(lr1, beta1=0.9, beta2=0.98, epsilon=1e-9)
    optimizer2 = tf.train.AdamOptimizer(lr2, beta1=0.9, beta2=0.98, epsilon=1e-9)    
    all_vars = tf.trainable_variables()
    grads = tf.gradients(model_ig.loss, all_vars)
    grads1 = grads[:-2]
    grads2 = grads[-2:]
    train_op1 = optimizer1.apply_gradients(zip(grads1, all_vars[:-2]), global_step=global_step1)
    train_op2 = optimizer2.apply_gradients(zip(grads2, all_vars[-2:]), global_step=global_step2)
    train_op = tf.group(train_op1, train_op2)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(data_pack):
        feed_dict = {
            model_ig.input_yg: data_pack.labelg,
            model_ig.input_yi: data_pack.labeli,
            model_ig.sequence_lengths: data_pack.sq_len_x,
            model_ig.dropout_keep_prob: 0.7,
            model_ig.memory_data:data_pack.memory_data,
            model_ig.memory_mask:data_pack.memory_mask,
            model_ig.memory_mask2:data_pack.memory_mask2,
            model_ig.memory_tag:data_pack.memory_tag,
            model_ig.memory_pos:data_pack.memory_pos,
            model_ig.sen_data:data_pack.sen_data,
            model_ig.sen_mask:data_pack.sen_mask,
            model_ig.sen_tag:data_pack.sen_tag,
            model_ig.hidden_dropout_prob: 0.1,
            model_ig.attention_probs_dropout_prob: 0.1,
            model_ig.sequence_lengths_reshape: data_pack.sequence_lengths_reshape,
            model_ig.inputs_one_row_index: data_pack.inputs_one_row_index,
            model_ig.inputs_reshape_index: data_pack.inputs_reshape_index,
            model_ig.inputs_reshape_shape: data_pack.inputs_reshape_shape,
            model_ig.tag_indices_reshape_shape: data_pack.tag_indices_reshape_shape,
            model_ig.one_column_head_index: data_pack.one_column_head_index,
            model_ig.one_column_head_matrix_index: data_pack.one_column_head_matrix_index,
            model_ig.one_column_pro: data_pack.one_column_pro,
            model_ig.one_column_pro_matrix_index: data_pack.one_column_pro_matrix_index,
            model_ig.one_column_matrix_shape: data_pack.one_column_matrix_shape,
            model_ig.not_one_column_index: data_pack.not_one_column_index,
            model_ig.has_known_tag: data_pack.has_known_tag,
            model_ig.has_skip_word: data_pack.has_skip_word, 
            model_ig.has_head_index: data_pack.has_head_index,
            model_ig.batch_size_reshape: data_pack.batch_size_reshape,         
        }
        _, step, loss, y_g, prediction = sess.run(
            [train_op, global_step1,
             model_ig.loss, model_ig.input_yg, model_ig.prediction],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))
        return loss, prediction

    def dev_step(data_pack):
        feed_dict = {
            model_ig.input_yg: data_pack.labelg,
            model_ig.input_yi: data_pack.labeli,
            model_ig.sequence_lengths: data_pack.sq_len_x,
            model_ig.dropout_keep_prob: 1.,
            model_ig.memory_data:data_pack.memory_data,
            model_ig.memory_mask:data_pack.memory_mask,
            model_ig.memory_mask2:data_pack.memory_mask2,
            model_ig.memory_tag:data_pack.memory_tag,
            model_ig.memory_pos:data_pack.memory_pos,
            model_ig.sen_data:data_pack.sen_data,
            model_ig.sen_mask:data_pack.sen_mask,
            model_ig.sen_tag:data_pack.sen_tag,
            model_ig.hidden_dropout_prob: 0.,
            model_ig.attention_probs_dropout_prob: 0.,
            model_ig.sequence_lengths_reshape: data_pack.sequence_lengths_reshape,
            model_ig.inputs_one_row_index: data_pack.inputs_one_row_index,
            model_ig.inputs_reshape_index: data_pack.inputs_reshape_index,
            model_ig.inputs_reshape_shape: data_pack.inputs_reshape_shape,
            model_ig.tag_indices_reshape_shape: data_pack.tag_indices_reshape_shape,
            model_ig.one_column_head_index: data_pack.one_column_head_index,
            model_ig.one_column_head_matrix_index: data_pack.one_column_head_matrix_index,
            model_ig.one_column_pro: data_pack.one_column_pro,
            model_ig.one_column_pro_matrix_index: data_pack.one_column_pro_matrix_index,
            model_ig.one_column_matrix_shape: data_pack.one_column_matrix_shape,
            model_ig.not_one_column_index: data_pack.not_one_column_index,
            model_ig.has_known_tag: data_pack.has_known_tag,
            model_ig.has_skip_word: data_pack.has_skip_word,
            model_ig.has_head_index: data_pack.has_head_index,
            model_ig.batch_size_reshape: data_pack.batch_size_reshape, 
        }
        step, loss, prediction, y_g, last_layer_att, transition_params1, transition_params2 = sess.run(
            [global_step1, model_ig.loss,
             model_ig.prediction, model_ig.input_yg, model_ig.last_layer_att, 
             model_ig.transition_params1, model_ig.transition_params2],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        return loss, prediction, last_layer_att, transition_params1, transition_params2

    for epoch in range(FLAGS.epochs):
        print('epoch: ', epoch)
        t_batch_list = prepare_batch(train_data, train_labelg, train_labeli, train_data_part, train_memoryg, train_memoryg_part, FLAGS.batch_size, FLAGS.num_tags, FLAGS.hidden_dim)
        train_list = list(t_batch_list)
        random.shuffle(train_list)
        t_batch_list = train_list
        recoder = open("recoder_gcrf_fill_trans_mem3_h512_b8_crfloss2_posfix_mean1_noinit_pretrain_olddata_L4H4_lr3e-5_trans_diff2X_part_dp7.txt","a")
        for i in range(0, len(t_batch_list)):
            loss_t, each_pred = train_step(t_batch_list[i])
            loss_train.append(loss_t)
            predictions = []
            labels = []
            for k in range(0, len(each_pred)):
                predictions.extend(each_pred[k,:t_batch_list[i].sq_len_x[k]])
                labels.extend(t_batch_list[i].labelg[k,:t_batch_list[i].sq_len_x[k]])   
            print(accuracy_score(labels, predictions))
            current_step = tf.train.global_step(sess, global_step1)
            if current_step % 250 == 0 and current_step>=60000:
                print('****************************************')
                print("\nEvaluation:")
                recoder.write("\nEvaluation*****************************")
                d_batch_list = prepare_batch(dev_data, dev_labelg, dev_labeli, dev_data_part, dev_memoryg, dev_memoryg_part, FLAGS.batch_size, FLAGS.num_tags, FLAGS.hidden_dim)
                predictions = []
                labels = []
                loss_d_temp = []
                for j in range(0, len(d_batch_list)):
                    loss_d, each_pred, _, t1, t2 = \
                            dev_step(d_batch_list[j])
                    loss_d_temp.append(loss_d)
                    for k in range(0, len(each_pred)):
                        predictions.extend(each_pred[k,:d_batch_list[j].sq_len_x[k]])
                        labels.extend(d_batch_list[j].labelg[k,:d_batch_list[j].sq_len_x[k]])
                loss_dev.append(sum(loss_d_temp)/len(loss_d_temp))
                recoder.write(str(current_step)+"\n")
                recoder.write(str(sum(loss_d_temp)/len(loss_d_temp))+"\n")
                print(accuracy_score(labels, predictions))
                recoder.write(str(accuracy_score(labels, predictions))+"\n")
                print(classification_report(labels, predictions, digits=4))
                recoder.write(str(classification_report(labels, predictions, digits=4))+"\n")
                print(confusion_matrix(labels, predictions))
                recoder.write(str(confusion_matrix(labels, predictions))+"\n")
                print("*******************************")
                recoder.write("*******************************"+"\n")

                print("\nTest: ")
                recoder.write("Test************************************")
                e_batch_list = prepare_batch(test_data, test_labelg, test_labeli, test_data_part, test_memoryg, test_memoryg_part, FLAGS.batch_size, FLAGS.num_tags, FLAGS.hidden_dim)

                test_input = []
                test_memory = []
                test_sentence_attention = []
                test_word_attention = []
                test_prediction = []
                test_label = []
                test_predictions = []
                test_labels = []
                for l in range(0, len(e_batch_list)):
                    _, test_each_pred, test_last_layer_att, transition_params1, transition_params2 = dev_step(e_batch_list[l])
                    for m in range(0, len(test_each_pred)):
                        test_predictions.extend(test_each_pred[m,:e_batch_list[l].sq_len_x[m]])
                        test_labels.extend(e_batch_list[l].labelg[m,:e_batch_list[l].sq_len_x[m]])
                    for d in range(0, len(test_each_pred)):
                        # test_input.append(e_x_batch[l][d,:e_seq_length[l][d]])
                        # test_memory.append(e_memory[l][d,:])
                        # test_sentence_attention.append(test_sentence_att[d,:e_seq_length[l][d],:])
                        # test_word_attention.append(test_word_att[d,:e_seq_length[l][d],:])
                        # test_prediction.append(test_each_pred[d,:e_seq_length[l][d]])
                        # test_label.append(e_y_batch[l][d,:e_seq_length[l][d]])
                        test_input.append(e_batch_list[l].sen_data[d,:e_batch_list[l].sq_len_x[d]])
                        test_memory.append(e_batch_list[l].memory_data[d])
                        test_sentence_attention.append(str(test_last_layer_att[d].tolist()))
                        test_prediction.append(test_each_pred[d,:e_batch_list[l].sq_len_x[d]])
                        test_label.append(e_batch_list[l].labelg[d,:e_batch_list[l].sq_len_x[d]])

                print(accuracy_score(test_labels, test_predictions))
                recoder.write(str(accuracy_score(test_labels, test_predictions))+"\n")
                weighted_f1 = get_weighted_f1(classification_report(test_labels, test_predictions, digits=4, output_dict=True))
                print("weighted f1: ", weighted_f1)
                recoder.write("weighted f1: " + str(weighted_f1) + '\n')
                print(classification_report(test_labels, test_predictions, digits=4))
                recoder.write(str(classification_report(test_labels, test_predictions, digits=4))+"\n")
                print(confusion_matrix(test_labels, test_predictions))
                recoder.write(str(confusion_matrix(test_labels, test_predictions))+"\n")

                # print(transition_params1)
                # print(transition_params2)
                # recoder.write(str(transition_params1)+"\n")
                # recoder.write(str(transition_params2)+"\n")

                print("\nTest Finished!!!")
                print("*******************************")
                recoder.write("*******************************"+"\n")

                # if(current_step == 50000):
                #     test_input_word = []
                #     test_memory_word = []
                #     for items_test_input in test_input:
                #         str_test_input = ""
                #         for number in items_test_input:
                #             get_index = dict_index.index(number)
                #             str_test_input = str_test_input + dict_word[get_index]+ " "
                #         test_input_word.append(str_test_input)

                #     for items_test_memory in test_memory:
                #         temp = []
                #         # for i in items_test_memory:
                #         #     str_test_memory = ""
                #         #     for index in i:
                #         #         get_index = dict_index.index(index)
                #         #         str_test_memory = (str_test_memory + dict_word[get_index]+ " ").replace("mask ","")
                #         #     temp.append(str_test_memory)
                #         str_test_memory = ""
                #         for index in items_test_memory:
                #             get_index = dict_index.index(index)
                #             str_test_memory = (str_test_memory + dict_word[get_index]+ " ").replace("mask ","")
                #         temp.append(str_test_memory)
                #         str_temp = "\n".join(temp)
                #         test_memory_word.append(str_temp)

                #     column1 = pd.Series(test_input_word, name='input')
                #     column2 = pd.Series(test_memory_word, name='memory')

                #     column3 = pd.Series(test_sentence_attention, name='attention')
                #     column4 = pd.Series(test_prediction, name='prediction')
                #     column5 = pd.Series(test_label, name='label')
                #     save = pd.concat([column1, column2, column3, column4, column5], axis=1)
                #     #save = pd.concat([column1, column4, column5], axis=1)
                #     save.to_csv("result_gcrf_fill_trans_mem3_h512_b8_crfloss_posfix_mean1_noinit_pretrain_olddata_L6H8_en1_lr3e.csv", encoding="gbk")
                # #     f = open("result_gcrf_nolinear_fill_trans_mem_h512_part_b8_posfix.txt", "a")

                # #     temp_count = 0
                # #     for case in test_word_attention:
                # #         str1 = str(temp_count) + "***" + "\n" + str(case) + "\n"
                # #         f.write(str1)
                # #         temp_count = temp_count + 1


print(len(loss_train))
print(len(loss_dev))
plt.switch_backend('agg')
plt.xlabel('batch', fontproperties='SimHei', fontsize=20)
plt.ylabel('loss', fontproperties='SimHei', fontsize=20)
plt.figure(1)
plt.plot(loss_train, color="green",marker=".", label="train")
loss_dev_x = [500+500*i for i in range(len(loss_dev))]
plt.plot(list(loss_dev_x),loss_dev, color="red", marker=".", label="dev")
plt.legend(loc='upper left')
plt.savefig('loss_em150_lr_modifymask_pan1e-3',dpi=800)
