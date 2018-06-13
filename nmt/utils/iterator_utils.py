# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
  pass


def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       src_max_len=None):
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids
  src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  # Add in the word counts.
  src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            0))  # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=None,
      target_output=None,
      source_sequence_length=src_seq_len,
      target_sequence_length=None)


def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True):
  if not output_buffer_size: # ？？？
    output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32) # 找到辅助单词的id
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset)) 
  # zip做的事情是将输入和目标一一对应起来构成数据集

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  # 这是要把数据切分开么，从而给多个电脑计算，前者参数是分成多少份，后者参数是返回哪一份（也就是电脑id）
  # Creates a Dataset that includes only 1/num_shards of this dataset.
  # This dataset operator is very useful when running distributed training, as it allows each worker to read a unique subset.
  if skip_count is not None: 
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)
    # Creates a Dataset that skips count elements from this dataset. 然而，为甚么要跳过一些样本呢？

  src_tgt_dataset = src_tgt_dataset.shuffle( # shuffle 数据集
      output_buffer_size, random_seed, reshuffle_each_iteration)
  # output_buffer_size: representing the number of elements from this dataset from which the new dataset will sample. 
  # 没懂，shuffle就shuffle呗，怎么还有采样的事情
  # reshuffle_each_iteration: 是否每个迭代都要shuffle，默认是True

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values), # 这是把句子切分成了单词
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # map(map_func, num_parallel_calls=None)
  # map_func: A function mapping a nested structure of tensors to another nested structure of tensors.
  # num_parallel_calls:  representing the number elements to process in parallel. If not specified, elements will be processed sequentially.
  # prefetch(output_buffer_size) Creates a Dataset that prefetches elements from this dataset.
  # output_buffer_size: representing the maximum number of elements that will be buffered when prefetching.
  # 具体来说，string_split() 返回一个SparseTensor实例，包含3个属性，indices、values、shape，
  # 意思是用更复杂也是省空间的方式保存一个稀疏的tensor，values是一个列表，这个地方的values应该是[word1, word2, ...]

  # Filter zero length input sequences. # 过滤掉零长样本
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  if src_max_len: # 处理超长样本，方式就是截取
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  # 单词字符串转id
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  # 二元组(src, tgt)变三元组(src, <sos>tgt, tgt<eos>)
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Add in sequence lengths.
  # 三元组(src, <sos>tgt, tgt<eos>)变五元组(src, <sos>tgt, tgt<eos>, len(src), len(tgt_in))
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      # 函数功能：给一个样本，返回其bucket id
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply( # 将数据进行 分组、reduce 处理
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
    # group_by_window(key_func, reduce_func, window_size)
    # 将数据按照键值分组，并reduce它们，？？？
    #   key_func: 生成键值的函数
    #   reduce_func: 
    # return: A Dataset transformation function.

  else:
    batched_dataset = batching_func(src_tgt_dataset)

  # 下面这坨代码的含义看起来像是，构造了一个迭代，然后返回了一组数据
  # 但是这个函数get_iterator不是应该返回迭代器么
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
   tgt_seq_len) = (batched_iter.get_next())
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)
