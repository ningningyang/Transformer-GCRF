# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs

class CrfDecodeForwardRnnCell(rnn_cell.RNNCell):
  """Computes the forward decoding in a linear-chain CRF.
  """

  def __init__(self, transition_params):
    """Initialize the CrfDecodeForwardRnnCell.
    Args:
      transition_params: A [num_tags, num_tags] matrix of binary
        potentials. This matrix is expanded into a
        [1, num_tags, num_tags] in preparation for the broadcast
        summation occurring within the cell.
    """
    self._transition_params = array_ops.expand_dims(transition_params, 0)
    self._num_tags = array_ops.shape(transition_params)[0]

  @property
  def state_size(self):
    return self._num_tags

  @property
  def output_size(self):
    return self._num_tags

  def __call__(self, inputs, state, scope=None):
    """Build the CrfDecodeForwardRnnCell.
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous step's
            score values.
      scope: Unused variable scope of this cell.
    Returns:
      backpointers: A [batch_size, num_tags] matrix of backpointers.
      new_state: A [batch_size, num_tags] matrix of new score values.
    """
    # For simplicity, in shape comments, denote:
    # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
    state = array_ops.expand_dims(state, 2)                         # [B, O, 1]

    # This addition op broadcasts self._transitions_params along the zeroth
    # dimension and state along the second dimension.
    # [B, O, 1] + [1, O, O] -> [B, O, O]
    transition_scores = state + self._transition_params             # [B, O, O]
    new_state = inputs + math_ops.reduce_max(transition_scores, [1])  # [B, O]
    backpointers = math_ops.argmax(transition_scores, 1)
    backpointers = math_ops.cast(backpointers, dtype=dtypes.int32)    # [B, O]
    return backpointers, new_state


class CrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
  """Computes backward decoding in a linear-chain CRF.
  """

  def __init__(self, num_tags):
    """Initialize the CrfDecodeBackwardRnnCell.
    Args:
      num_tags: An integer. The number of tags.
    """
    self._num_tags = num_tags

  @property
  def state_size(self):
    return 1

  @property
  def output_size(self):
    return 1

  def __call__(self, inputs, state, scope=None):
    """Build the CrfDecodeBackwardRnnCell.
    Args:
      inputs: A [batch_size, num_tags] matrix of
            backpointer of next step (in time order).
      state: A [batch_size, 1] matrix of tag index of next step.
      scope: Unused variable scope of this cell.
    Returns:
      new_tags, new_tags: A pair of [batch_size, num_tags]
        tensors containing the new tag indices.
    """
    state = array_ops.squeeze(state, axis=[1])                # [B]
    batch_size = array_ops.shape(inputs)[0]
    b_indices = math_ops.range(batch_size)                    # [B]
    # batch_size = tf.Print(batch_size, [batch_size], "batch_size", summarize=200)
    # b_indices = tf.Print(b_indices, [b_indices], "b_indices", summarize=200)
    indices = array_ops.stack([b_indices, state], axis=1)     # [B, 2]
    new_tags = array_ops.expand_dims(
        gen_array_ops.gather_nd(inputs, indices),             # [B]
        axis=-1)                                              # [B, 1]
    return new_tags, new_tags

def crf_decode(inputs, 
                transition_params1, 
                transition_params2, 
                sequence_lengths_reshape,#分成短句后，每句的长度
                inputs_one_row_index,#原input中非padding的位置
                inputs_reshape_index,#原input中每个词放到分成短句后矩阵的位置
                inputs_reshape_shape,#分成短句后的矩阵大小 ？*？*numtag
                tag_indices_reshape_shape,#分成短句后的tag的矩阵大小 ？*？
                one_column_head_index,#句首需要作为未知tag被取出来算竖链的行编号
                one_column_head_matrix_index,#未知tag放入竖链的位置
                one_column_pro,#分成短句后出现的句首已知tag
                one_column_pro_matrix_index,#已知tag放入竖链的位置
                one_column_matrix_shape,#竖链的矩阵大小 ？*numtag
                not_one_column_index,#没有作为未知tag取出放到竖链的行编号
                has_known_tag,
                has_skip_word,
                has_head_index,
                batch_size_reshape,
            ):
    """Decode the highest scoring sequence of tags in TensorFlow.
    This is a function for tensor.
    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
      transition_params: A [num_tags, num_tags] matrix of
                binary potentials.
      sequence_length: A [batch_size] vector of true sequence lengths.
      skip_chain: [batch_size, max_seq_len]
    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                  Contains the highest scoring tag indices.
      best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """
    origin_batch_size = inputs.get_shape().as_list()[0]
    origin_sequence_lenth = inputs.get_shape().as_list()[1]
    num_tags = inputs.get_shape().as_list()[2]
    reshape_batch_size = sequence_lengths_reshape.get_shape().as_list()[0]

    # reshape inputs
    inputs_one_row = tf.gather_nd(inputs, inputs_one_row_index)
    inputs_reshape = tf.scatter_nd(inputs_reshape_index, inputs_one_row, inputs_reshape_shape)
    inputs_reshape = gen_array_ops.reverse_sequence(inputs_reshape, sequence_lengths_reshape, seq_dim=1)
    inputs_reshape = tf.reshape(inputs_reshape, shape=[tf.shape(sequence_lengths_reshape)[0], -1, inputs.get_shape()[2]])

    # known tags
    one_column_pro_bitmap = tf.cast(tf.one_hot(one_column_pro, num_tags), dtype=bool)

    # forward
    ## linear
    linear_initial_state = array_ops.slice(inputs_reshape, [0, 0, 0], [-1, 1, -1])
    linear_initial_state = array_ops.squeeze(linear_initial_state, [1])
    rest_of_linear_inputs = array_ops.slice(inputs_reshape, [0, 1, 0], [-1, -1, -1])
    sequence_lengths_less_one = math_ops.maximum(
            constant_op.constant(0, dtype=sequence_lengths_reshape.dtype),
            sequence_lengths_reshape - 1)
    linear_forward_cell = CrfDecodeForwardRnnCell(transition_params1)
    linear_backpointers, linear_last_column_score = rnn.dynamic_rnn(
                                cell=linear_forward_cell,
                                inputs=rest_of_linear_inputs,
                                sequence_length=sequence_lengths_less_one,
                                initial_state=linear_initial_state,
                                dtype=dtypes.int32)
    linear_backpointers = gen_array_ops.reverse_sequence(
            linear_backpointers, sequence_lengths_less_one, seq_dim=1)

    ## column
    def final_column_init():
        column_inputs_one_row = tf.gather_nd(linear_last_column_score, one_column_head_index)
        column_inputs = tf.scatter_nd(one_column_head_matrix_index, column_inputs_one_row, one_column_matrix_shape) 

        def column_inputs_add(column_inputs):
            filtered_one_column_pro = array_ops.where(
                                        one_column_pro_bitmap, tf.zeros(array_ops.shape(one_column_pro_bitmap)),
                                        array_ops.fill(array_ops.shape(one_column_pro_bitmap), float("-inf")))
            pro_inputs = tf.scatter_nd(one_column_pro_matrix_index, filtered_one_column_pro, one_column_matrix_shape)
            column_inputs = column_inputs + pro_inputs
            return column_inputs
        column_inputs = tf.cond(has_known_tag, \
                                lambda: column_inputs_add(column_inputs), \
                                lambda: column_inputs)
        def _multi_seq_fn():
            column_inputs.set_shape([None, num_tags])
            column_sequence_length = tf.expand_dims(tf.shape(column_inputs)[0] , 0)
            column_sequence_length_less_one = tf.expand_dims(tf.shape(column_inputs)[0] - 1, 0)
            column_initial_state = array_ops.expand_dims(column_inputs[0], 0)
            column_rest_inputs = array_ops.expand_dims(column_inputs[1:], 0)
            column_forward_cell = CrfDecodeForwardRnnCell(transition_params2)
            column_backpointers, final_score = rnn.dynamic_rnn(
                    cell=column_forward_cell,
                    inputs=column_rest_inputs,
                    sequence_length=column_sequence_length_less_one,
                    initial_state=column_initial_state,
                    time_major=False,
                    dtype=dtypes.int32)
            column_backpointers = gen_array_ops.reverse_sequence( 
                    column_backpointers, column_sequence_length_less_one, seq_dim=1)        

            #decode
            crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
            column_initial_state = math_ops.cast(math_ops.argmax(final_score, axis=1),  # [B]
                                      dtype=dtypes.int32)
            column_initial_state = array_ops.expand_dims(column_initial_state, axis=-1)  # [B, 1]
            column_decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, 1]
                cell=crf_bwd_cell,
                inputs=column_backpointers,
                sequence_length=column_sequence_length_less_one,
                initial_state=column_initial_state,
                time_major=False,
                dtype=dtypes.int32)
            column_decode_tags = array_ops.squeeze(column_decode_tags, axis=[2])  # [B, T - 1]
            column_decode_tags = array_ops.concat([column_initial_state, column_decode_tags],   # [B, T]
                                       axis=1)
            column_decode_tags = gen_array_ops.reverse_sequence(  # [B, T]
                column_decode_tags, column_sequence_length, seq_dim=1)      

            column_decode_tags = array_ops.squeeze(column_decode_tags, axis=[0])
            column_head_tag_one_row = tf.gather_nd(column_decode_tags, one_column_head_matrix_index)
            final_column = tf.scatter_nd(one_column_head_index, column_head_tag_one_row, tf.shape(sequence_lengths_reshape))
            final_column = tf.reshape(final_column, shape=[batch_size_reshape])
            return final_column

        def _single_seq_fn():
            return math_ops.cast(math_ops.argmax(column_inputs, axis=1),  # [B]
                                  dtype=dtypes.int32)

        final_column = utils.smart_cond(
                          pred=math_ops.equal(
                             array_ops.shape(column_inputs)[0], 1),
                          true_fn=_single_seq_fn,
                          false_fn=_multi_seq_fn)

        return final_column

    final_column = tf.zeros([batch_size_reshape], dtype=dtypes.int32)
    final_column = tf.cond(has_head_index, \
                        lambda: final_column_init(), \
                        lambda: final_column)

    def final_column_add(final_column):
        column_origin_tag = math_ops.cast(math_ops.argmax(linear_last_column_score, axis=1),  # [B]
                                  dtype=dtypes.int32)
        column_pro_tag_one_row = tf.gather_nd(column_origin_tag, not_one_column_index)
        column_pro_tag = tf.scatter_nd(not_one_column_index, column_pro_tag_one_row, tf.shape(sequence_lengths_reshape))
        final_column = final_column + column_pro_tag
        return final_column
    final_column = tf.cond(has_skip_word, \
                            lambda: final_column_add(final_column), \
                            lambda: final_column)

    linear_decode_initial_state =  array_ops.expand_dims(final_column,  -1)
    crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
    linear_decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, 1]
                                    cell=crf_bwd_cell,
                                    inputs=linear_backpointers,
                                    sequence_length=sequence_lengths_less_one,
                                    initial_state=linear_decode_initial_state,
                                    time_major=False,
                                    dtype=dtypes.int32)
    linear_decode_tags = array_ops.squeeze(linear_decode_tags, axis=[2])
    linear_decode_tags = array_ops.concat([linear_decode_initial_state, linear_decode_tags],   # [B, T]
                                   axis=1) 
    # linear_decode_tags = gen_array_ops.reverse_sequence(  # [B, T]
    #         linear_decode_tags, sequence_lengths_reshape, seq_dim=1)

    # reshape back tags
    decode_tags_one_row = tf.gather_nd(linear_decode_tags, inputs_reshape_index)
    final_decode_tag = tf.scatter_nd(inputs_one_row_index, decode_tags_one_row, tf.shape(inputs)[:2])
    # final_decode_tag = gen_array_ops.reverse_sequence(final_decode_tag, sequence_lengths_reshape, seq_dim=1)
    return final_decode_tag

#**********************************test********************************
# input = np.array([[[3, 2], [2, 1]], [[2, 4], [2, 1]], [[3, 2], [1, 1]]])
# input_t = tf.constant(input, dtype=tf.float32)
# tag_t = tf.constant(np.array([[0, 1], [0, 1], [1, 0]]), dtype=tf.int64)
# len_t = tf.constant(np.array([2, 2, 2]), dtype=tf.int64)
# trans_t = tf.constant(np.array([[1, 2], [2, 3]]), dtype=tf.float32)

# inputs_one_row_index = tf.constant(np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]), dtype=tf.int64)
# inputs_reshape_index = tf.constant(np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]), dtype=tf.int64)
# inputs_reshape_shape = tf.constant(np.array([3,2,2]), dtype=tf.int64)
# tag_indices_reshape_shape = tf.constant(np.array([3,2]), dtype=tf.int64)
# one_column_head_index = tf.constant(np.array([[0],[2]]), dtype=tf.int64)
# one_column_head_matrix_index = tf.constant(np.array([[0],[2]]), dtype=tf.int64)
# one_column_pro = tf.constant(np.array([1]), dtype=tf.int64)
# one_column_pro_matrix_index = tf.constant(np.array([[1],]), dtype=tf.int64)
# # one_column_pro = tf.constant(np.array([]), dtype=tf.int64)
# # one_column_pro_matrix_index = tf.constant(np.array([]), dtype=tf.int64)
# one_column_matrix_shape = tf.constant(np.array([3,2]), dtype=tf.int64)
# not_one_column_index = tf.constant(np.array([[1],]), dtype=tf.int64)
# # not_one_column_index = tf.constant(np.array([]), dtype=tf.int64)


# with tf.Session() as sess:
#     decode_tags = sess.run(
#                     crf_decode(
#                         input_t,
#                         trans_t,
#                         trans_t,
#                         len_t,
#                         inputs_one_row_index,
#                         inputs_reshape_index,
#                         inputs_reshape_shape,
#                         tag_indices_reshape_shape,
#                         one_column_head_index,
#                         one_column_head_matrix_index,
#                         one_column_pro,                                                
#                         one_column_pro_matrix_index,
#                         one_column_matrix_shape,
#                         not_one_column_index,
#                         )
#                     )
#     print("decode_tags:", decode_tags)