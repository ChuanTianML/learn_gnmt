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

"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf

from tensorflow.python.layers import core as layers_core

from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]


class BaseModel(object):
  """Sequence-to-sequence base class.
  """
  

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      reverse_target_vocab_table: Lookup table mapping ids to target words. Only
        required in INFER mode. Defaults to None.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.

    """
    assert isinstance(iterator, iterator_utils.BatchedInput)
    # BatchedInput 返回Coordinate类，属性为5元组数据和一个iterator初始化器
    # 所以，其实iterator并不是一个迭代器，而是由迭代器获取的一个batch的样本
    self.iterator = iterator
    self.mode = mode
    self.src_vocab_table = source_vocab_table	# src 词表
    self.tgt_vocab_table = target_vocab_table 	# tgt 词表

    self.src_vocab_size = hparams.src_vocab_size # 词表大小
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.num_gpus = hparams.num_gpus # gpu数量
    self.time_major = hparams.time_major

    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None # ？？？
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Set num layers
    self.num_encoder_layers = hparams.num_encoder_layers # encoder和decoder的层数
    self.num_decoder_layers = hparams.num_decoder_layers
    assert self.num_encoder_layers # 至少一层
    assert self.num_decoder_layers

    # Set num residual layers
    # 这个if else 逻辑，应该是说如果没分别指定encoder和decoder的多余连接层数，则认为两者使用相同的多余连接层数
    if hasattr(hparams, "num_residual_layers"):  # compatible common_test_utils 
      self.num_encoder_residual_layers = hparams.num_residual_layers
      self.num_decoder_residual_layers = hparams.num_residual_layers
    else:
      self.num_encoder_residual_layers = hparams.num_encoder_residual_layers
      self.num_decoder_residual_layers = hparams.num_decoder_residual_layers

    # Initializer
    initializer = model_helper.get_initializer( # 获得初始化器
        hparams.init_op, hparams.random_seed, hparams.init_weight)
    ## init_op指定初始化方式(均匀分布、正态分布)，后面两个参数分别是种子和范围。
    tf.get_variable_scope().set_initializer(initializer) # 这一步应该是用initializer初始化各个变量吧？？？
    # tf.get_variable_scope()： Returns the current variable scope.

    # Embeddings
    self.init_embeddings(hparams, scope) # 初始化encoder和decoder的embedding
    self.batch_size = tf.size(self.iterator.source_sequence_length) 
    # 一个batch的source_sequence_length的大小，正好就是batch_size

    # Projection
    # 命名空间用来管理变量
    with tf.variable_scope(scope or "build_network"):
      with tf.variable_scope("decoder/output_projection"):
        self.output_layer = layers_core.Dense( 
            hparams.tgt_vocab_size, use_bias=False, name="output_projection")
       	
       	"""	这一步构造一个全连接的网络，Dense是一个类
       		其构造函数__init__(self, units)中的units：Integer or Long, dimensionality of the output space.
       	"""

    ## Train graph
    res = self.build_graph(hparams, scope=scope)
    # build_graph() return logits, loss, final_context_state, sample_id

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1] # 交叉熵损失
      self.word_count = tf.reduce_sum( # 单词数量是source 和 target 单词数量之和
          self.iterator.source_sequence_length) + tf.reduce_sum(
              self.iterator.target_sequence_length)
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      # 从这里看，sample_id 是word id，用sample_id 找到单词，完成翻译
      self.infer_logits, _, self.final_context_state, self.sample_id = res
      self.sample_words = reverse_target_vocab_table.lookup(
          tf.to_int64(self.sample_id))

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum( # 预测出的单词的数量
          self.iterator.target_sequence_length)

    self.global_step = tf.Variable(0, trainable=False)
    params = tf.trainable_variables() # 所有的变量

    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(hparams.learning_rate)
      # warm-up
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

      # Optimizer # 根据参数设定寻优方式
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        tf.summary.scalar("lr", self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)

      # Gradients
      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

      clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.grad_norm = grad_norm

      self.update = opt.apply_gradients(
          zip(clipped_grads, params), global_step=self.global_step)

      # Summary
      self.train_summary = tf.summary.merge([
          tf.summary.scalar("lr", self.learning_rate),
          tf.summary.scalar("train_loss", self.train_loss),
      ] + grad_norm_summary)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

    # Saver
    self.saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

    # Print trainable variables
    utils.print_out("# Trainable variables")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(
          tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warump_cond")

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
      decay_factor = 0.5
      if hparams.decay_scheme == "luong5":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 5
      elif hparams.decay_scheme == "luong10":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 10
      elif hparams.decay_scheme == "luong234":
        start_decay_step = int(hparams.num_train_steps * 2 / 3)
        decay_times = 4
      remain_steps = hparams.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / decay_times)
    elif not hparams.decay_scheme:  # no decay
      start_decay_step = hparams.num_train_steps
      decay_steps = 0
      decay_factor = 1.0
    elif hparams.decay_scheme:
      raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
    utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                    "decay_factor %g" % (hparams.decay_scheme,
                                         start_decay_step,
                                         decay_steps,
                                         decay_factor))

    return tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(
            self.learning_rate,
            (self.global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")

  def init_embeddings(self, hparams, scope):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = ( 		# 获得encoder和decoder的embedding
        model_helper.create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=hparams.num_units,
            tgt_embed_size=hparams.num_units,
            num_partitions=hparams.num_embeddings_partitions,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file,
            scope=scope,))

  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    return sess.run([self.update,
                     self.train_loss,
                     self.predict_count,
                     self.train_summary,
                     self.global_step,
                     self.word_count,
                     self.batch_size,
                     self.grad_norm,
                     self.learning_rate])

  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([self.eval_loss,
                     self.predict_count,
                     self.batch_size])

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss, final_context_state),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols]. 	# ？？？
        loss: the total loss / batch_size.							# 损失
        final_context_state: The final state of decoder RNN.		# decoder的最终状态

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
      # Encoder
      encoder_outputs, encoder_state = self._build_encoder(hparams) # build encoder 图，输出了encoder的输出、状态

      	""" 如果是单向网络：
		    encoder_outputs:
		    	If time_major == False: [batch_size, max_time, cell.output_size]
		    	If time_major == True:  [max_time, batch_size, cell.output_size]
		    encoder_state:
		    	If cell.state_size is an int:        [batch_size, cell.state_size] # 相比output，没有了timestep
		    	If cell.state_size is a TensorShape: [batch_size] + cell.state_size
   		如果是双向网络：
	   		encoder_outputs:
	   			If time_major == False: [batch_size, max_time, cell.fw_output_size + cell.bw_output_size]
		    	If time_major == True:  [max_time, batch_size, cell.output_size  + cell.bw_output_size]
	   		encoder_state: 
	   			if num_bi_layers == 1: (output_state_fw, output_state_bw)
					output_state_fw: [batch_size, cell_fw.output_state_size]
					output_state_bw: [batch_size, cell_bw.output_state_size]
				else: 
    	"""

      ## Decoder
      logits, sample_id, final_context_state = self._build_decoder( # build decoder 图，输出了
          encoder_outputs, encoder_state, hparams)
      # logits： 未归一化的概率，也就是模型的输出
      # sample_id： ？？？

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER: # 如果不是测试模式
        with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1, # 这是做什么
                                                   self.num_gpus)):
          loss = self._compute_loss(logits) # 根据模型的输出，计算损失，交叉熵损失
      else:
        loss = None

      return logits, loss, final_context_state, sample_id

  @abc.abstractmethod
  def _build_encoder(self, hparams): # 建encoder的图，抽象函数
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                          base_gpu=0):
    """Build a multi-layer RNN cell that can be used by encoder."""

    # 返回的是单步time step的多个神经元，数量是神经元层数
    return model_helper.create_rnn_cell( 
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        base_gpu=base_gpu,
        single_cell_fn=self.single_cell_fn)

  def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
    """Maximum decoding steps at inference time.""" 
    # 测试时的最大decode步数（time step），就是翻译一个多长的句子
    if hparams.tgt_max_len_infer: # 如果参数中有设置，那就使用参数设置
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else: # 如果没有参数设置，则用source句子的最长程度除以2作为翻译长度
      # TODO(thangluong): add decoding_length_factor flag
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(source_sequence_length)
      maximum_iterations = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step. # 每个time step的输出
      encoder_state: The final state of the encoder. # encoder的最终状态
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)), # 辅助词id
                         tf.int32)
    tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                         tf.int32)
    iterator = self.iterator

    # maximum_iteration: The maximum decoding steps.
    # 也就是要翻译一个多长的句子
    maximum_iterations = self._get_infer_maximum_iterations( # decode 最大步数，inference time是什么
        hparams, iterator.source_sequence_length)

      	""" encoder的输出
      		如果是单向网络：
			    encoder_outputs:
			    	If time_major == False: [batch_size, max_time, cell.output_size]
			    	If time_major == True:  [max_time, batch_size, cell.output_size]
			    encoder_state:
			    	If cell.state_size is an int:        [batch_size, cell.state_size] # 相比output，没有了timestep
			    	If cell.state_size is a TensorShape: [batch_size] + cell.state_size
			如果是双向网络：
		   		encoder_outputs:
		   			If time_major == False: [batch_size, max_time, cell.fw_output_size + cell.bw_output_size]
			    	If time_major == True:  [max_time, batch_size, cell.output_size  + cell.bw_output_size]
		   		encoder_state: 
		   			if num_bi_layers == 1: (output_state_fw, output_state_bw)
						output_state_fw: [batch_size, cell_fw.output_state_size]
						output_state_bw: [batch_size, cell_bw.output_state_size]
					else: 
		"""

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope: # 构造decoder网络
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          iterator.source_sequence_length)

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER: # 训练和验证模式
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = iterator.target_input # 获取输入
        if self.time_major:
          target_input = tf.transpose(target_input) # 如果需要，转化为time major
        decoder_emb_inp = tf.nn.embedding_lookup( # 进行embedding
            self.embedding_decoder, target_input)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper( # 这个helper是干啥使的？？？
            decoder_emb_inp, iterator.target_sequence_length,
            time_major=self.time_major)
        # A helper for use during training. Only reads inputs.
        # helper应该就是负责输入数据的
		# Returned sample_ids are the argmax of the RNN output logits.

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder( # TensorFlow内置的decoder
            cell, # decoder cell
            helper,
            decoder_initial_state,)

        # Dynamic decoding
        """	Perform dynamic decoding with decoder.
			Calls initialize() once and step() repeatedly on the Decoder object.
        """
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)
        # 这个outputs的具体内容是什么？？？

        sample_id = outputs.sample_id

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        logits = self.output_layer(outputs.rnn_output) 
        # 经过全连接网络层，并经过激活，得到在词典上的概率分布
        # 上网查，发现这个logits是未归一化的概率

      ## Inference
      else:
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        if beam_width > 0: ## 如果用beamsearch
          ### 使用TensorFlow内置的带有beamsearch的decoder
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder( 
              cell=cell,
              embedding=self.embedding_decoder, # A callable that takes a vector tensor of ids (argmax ids), or the params argument for embedding_lookup
              start_tokens=start_tokens, # int32 vector shaped [batch_size], the start tokens.
              end_token=end_token, # int32 scalar, the token that marks end of decoding
              initial_state=decoder_initial_state, 
              beam_width=beam_width, #Python integer, the number of beams.
              output_layer=self.output_layer, #  An instance of tf.layers.Layer, i.e., tf.layers.Dense. Optional layer to apply to the RNN output prior to storing the result or sampling.
              length_penalty_weight=length_penalty_weight) # Float weight to penalize length. Disabled with 0.0.
        else: ## 如果不用beamsearch，
          # Helper
          sampling_temperature = hparams.sampling_temperature
          """
          float32 scalar, value to divide the logits by before computing the softmax. 
          Larger values (above 1.0) result in more random samples, while smaller values push the sampling distribution towards the argmax. 
          Must be strictly greater than 0. Defaults to 1.0.
          """
          if sampling_temperature > 0.0:
            helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                self.embedding_decoder, start_tokens, end_token,
                softmax_temperature=sampling_temperature, # 这个参数的作用知晓了，但是具体是怎么采样的呢？？？
                seed=hparams.random_seed)
          else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper( # 最原始的贪婪算法，每个time step都输出概率最大的那个单词
                self.embedding_decoder, start_tokens, end_token)

          # Decoder
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_initial_state,
              output_layer=self.output_layer  # applied per timestep
          )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        if beam_width > 0:
          logits = tf.no_op() # Does nothing. Only useful as a placeholder for control edges.
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

    return logits, sample_id, final_context_state
    # logits: encoder之后再词典上的概率分布
    # sample_id: ？？？
    # final_context_state: 应该是decode之后的output_state

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Subclass must implement this.

    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      source_sequence_length: sequence length of encoder_outputs.

    Returns:
      A tuple of a multi-layer RNN cell used by decoder # 返回一个RNN神经单元，供decoder使用
        and the intial state of the decoder RNN.
    """
    pass

  def _compute_loss(self, logits):
    """Compute optimization loss."""
    target_output = self.iterator.target_output # 翻译的标准答案
    if self.time_major:
      target_output = tf.transpose(target_output) 
    max_time = self.get_max_time(target_output) # 标准翻译句子的长度，也就是time step的数量
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    # return: A Tensor of the same shape as labels and of the same type as logits with the softmax cross entropy loss.
    # label:  [batch_size, max_time]
    # logits: [batch_size, max_time, num_classes]
    # 两个参数：
    # 前者是label，也就是每个单词的id，后者是模型输出
    # 根据此计算交叉熵损失

    # 下面计算一个mask，意思是取标准答案长度范围内的损失
    # tf.sequence_mask([1, 3, 2], 5)
    # [[True, False, False, False, False],
    # [True, True, True, False, False],
    # [True, True, False, False, False]]
    target_weights = tf.sequence_mask(
        self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
    if self.time_major:
      target_weights = tf.transpose(target_weights)

    loss = tf.reduce_sum(
        crossent * target_weights) / tf.to_float(self.batch_size)
    return loss

  def _get_infer_summary(self, hparams):
    return tf.no_op() # 能干啥？
    # Does nothing. Only useful as a placeholder for control edges.
    # Args:	name: A name for the operation (optional).
	# Return: The created Operation.

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    return sess.run([
        self.infer_logits, self.infer_summary, self.sample_id, self.sample_words
    ])

  def decode(self, sess):
    """Decode a batch.

    Args:
      sess: tensorflow session to use.

    Returns:
      A tuple consiting of outputs, infer_summary.
        outputs: of size [batch_size, time]
    """
    _, infer_summary, _, sample_words = self.infer(sess)

    # make sure outputs is of shape [batch_size, time] or [beam_width,
    # batch_size, time] when using beam search.
    if self.time_major:
      sample_words = sample_words.transpose()
    elif sample_words.ndim == 3:  # beam search output in [batch_size,
                                  # time, beam_width] shape.
      sample_words = sample_words.transpose([2, 0, 1])
    return sample_words, infer_summary


class Model(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """

  def _build_encoder(self, hparams):
    """Build an encoder."""
    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers
    iterator = self.iterator

    source = iterator.source
    if self.time_major:
      source = tf.transpose(source)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype
      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      encoder_emb_inp = tf.nn.embedding_lookup( # 在embedding矩阵中lookup，完成数据转换
          self.embedding_encoder, source)

      # Encoder_outputs: [max_time, batch_size, num_units]
      if hparams.encoder_type == "uni": # 如果是单向网络
        utils.print_out("  num_layers = %d, num_residual_layers=%d" % # 网络形状告知
                        (num_layers, num_residual_layers))
        cell = self._build_encoder_cell( # 构建encoder神经单元
            hparams, num_layers, num_residual_layers)
        # 返回的是单步time step的多个神经元，数量是神经元层数，也就是说这个cell是一步的cell，而不是一个神经元


        # encoder_outputs： [source_sequence_length, dimension] 推测的
        # encoder_state:	[num_layers, dimension]
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn( # 使用TensorFlow API构建单向网络
            cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=iterator.source_sequence_length,
            time_major=self.time_major,
            swap_memory=True)
      elif hparams.encoder_type == "bi": # 如果是双向网络
        num_bi_layers = int(num_layers / 2) # 总层数除2是双向网络层数
        num_bi_residual_layers = int(num_residual_layers / 2) # 一层双向网络内部，不再做额外连接
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" % # 网络形状告知
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = ( # 构建双向网络
            self._build_bidirectional_rnn(
                inputs=encoder_emb_inp,
                sequence_length=iterator.source_sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))
		    """
		    (outputs, output_states) 
			    outputs:  (output_fw, output_bw) 
			    	output_fw: 
			    		if False == time_major: [batch_size, max_time, cell_fw.output_size]
			    		if True  == time_major: [max_time, batch_size, cell_fw.output_size]
			    	output_bw:
			    		if False == time_major: [batch_size, max_time, cell_bw.output_size]
			    		if True  == time_major: [max_time, batch_size, cell_bw.output_size]
			   	output_states: (output_state_fw, output_state_bw)  
			   		output_state_fw: [num_bi_layers, batch_size, cell_fw.output_state_size] 这形状是我推测的
			   		output_state_bw: [nun_bi_layers, batch_size, cell_bw.output_state_size]
		    """

        if num_bi_layers == 1: # 如果只有一个双向网络，
          encoder_state = bi_encoder_state
        else: # 如果多于1个双向网络，则将所有层的双向网络中的正向&反向的最后状态，拼成一个tuple作为输出
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else: # 除了单向和双向之外，不认识的网络结构参数，报错
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
    return encoder_outputs, encoder_state
    """ 如果是单向网络：
		    encoder_outputs:
		    	If time_major == False: [batch_size, max_time, cell.output_size]
		    	If time_major == True:  [max_time, batch_size, cell.output_size]
		    encoder_state:
		    	If cell.state_size is an int:        [batch_size, cell.state_size] # 相比output，没有了timestep
		    	If cell.state_size is a TensorShape: [batch_size] + cell.state_size
	   	如果是双向网络：
	   		encoder_outputs:
	   			If time_major == False: [batch_size, max_time, cell.fw_output_size + cell.bw_output_size]
		    	If time_major == True:  [max_time, batch_size, cell.output_size  + cell.bw_output_size]
	   		encoder_state: 
	   			if num_bi_layers == 1: (output_state_fw, output_state_bw)
					output_state_fw: [batch_size, cell_fw.output_state_size]
					output_state_bw: [batch_size, cell_bw.output_state_size]
				else: 
    """

  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`.
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams, # 构建正向神经网络单元
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(hparams, # 构建反向神经网络单元
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn( # 使用TensorFlow API构建双向网络
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=self.time_major,
        swap_memory=True)
    """
    (outputs, output_states) 
	    outputs:  (output_fw, output_bw) 
	    	output_fw: 
	    		if False == time_major: [batch_size, max_time, cell_fw.output_size]
	    		if True  == time_major: [max_time, batch_size, cell_fw.output_size]
	    	output_bw:
	    		if False == time_major: [batch_size, max_time, cell_bw.output_size]
	    		if True  == time_major: [max_time, batch_size, cell_bw.output_size]
	   	output_states: (output_state_fw, output_state_bw)  
	   		output_state_fw: [num_bi_layers, batch_size, cell_fw.output_state_size] 这形状是我推测的
	   		output_state_bw: [nun_bi_layers, batch_size, cell_bw.output_state_size]
    """

    return tf.concat(bi_outputs, -1), bi_state 
    # bi_outputs 经过concat之后，最后一维被拼接，即正反向输出被拼接
    # 以time_major=False为例，拼接后的outputs： [batch_size, max_time, cell_fw.output_size + cell_bw.output_size]
    # 最终返回的bi_state: 见上面

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models

  	""" encoder的输出
  		如果是单向网络：
		    encoder_outputs:
		    	If time_major == False: [batch_size, max_time, cell.output_size]
		    	If time_major == True:  [max_time, batch_size, cell.output_size]
		    encoder_state:
		    	If cell.state_size is an int:        [batch_size, cell.state_size] # 相比output，没有了timestep
		    	If cell.state_size is a TensorShape: [batch_size] + cell.state_size
		如果是双向网络：
	   		encoder_outputs:
	   			If time_major == False: [batch_size, max_time, cell.fw_output_size + cell.bw_output_size]
		    	If time_major == True:  [max_time, batch_size, cell.output_size  + cell.bw_output_size]
	   		encoder_state: 
	   			if num_bi_layers == 1: (output_state_fw, output_state_bw)
					output_state_fw: [batch_size, cell_fw.output_state_size]
					output_state_bw: [batch_size, cell_bw.output_state_size]
				else: 
	"""

    if hparams.attention:
      raise ValueError("BasicModel doesn't support attention.")

    # 同理于encoder，返回单步的一系列神经元，数量为网络层数
    cell = model_helper.create_rnn_cell( 
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=self.num_decoder_layers,
        num_residual_layers=self.num_decoder_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=self.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # For beam search, we need to replicate encoder infos beam_width times
    if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state
    """
    	构造decoder cell唯一需要encoder输出的地方是，初始化decoder_state用的是 encoder_state；
    	encoder_outputs在这里没有用到。
    """
