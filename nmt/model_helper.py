"""Utility functions for building models."""
from __future__ import print_function

import collections
import six
import os
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import vocab_utils


__all__ = [
    "get_initializer", "get_device_str", "create_train_model",
    "create_eval_model", "create_infer_model",
    "create_emb_for_encoder_and_decoder", "create_rnn_cell", "gradient_clip",
    "create_or_load_model", "load_model", "avg_checkpoints",
    "compute_perplexity"
]

# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 50000


def get_initializer(init_op, seed=None, init_weight=None):
  """Create an initializer. init_weight is only for uniform."""
  if init_op == "uniform": # 用户指定范围的均匀分布
    assert init_weight
    return tf.random_uniform_initializer(
        -init_weight, init_weight, seed=seed)
  elif init_op == "glorot_normal": # TensorFlow内置自生成范围的均匀分布
    return tf.keras.initializers.glorot_normal( 
        seed=seed)
  elif init_op == "glorot_uniform": # TensorFlow内置自生成范围的正太分布
    return tf.keras.initializers.glorot_uniform(
        seed=seed)
  else:
    raise ValueError("Unknown init_op %s" % init_op)


def get_device_str(device_id, num_gpus):
  """Return a device string for multi-GPU setup."""
  if num_gpus == 0:
    return "/cpu:0"
  device_str_output = "/gpu:%d" % (device_id % num_gpus)
  return device_str_output


class ExtraArgs(collections.namedtuple(
    "ExtraArgs", ("single_cell_fn", "model_device_fn",
                  "attention_mechanism_fn"))):
  pass


# 个人理解，创建一个只有key，还没有设置value的字典
class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                          "skip_count_placeholder"))):
  pass


def create_train_model(
    model_creator, hparams, scope=None, num_workers=1, jobid=0,
    extra_args=None):
  """Create train graph, model, and iterator."""
  # 函数功能： 创建训练图，模型和迭代器

  # 数据文件和词表文件
  src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
  tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "train"):
    # 读取词表文件到table（word2id）
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables( 
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)

    # 加载数据（看此逻辑，两个文件按行对应）
    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

    # 创建数据迭代器
    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets, # ？？？
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        skip_count=skip_count_placeholder, # ？？？
        num_shards=num_workers, # ？？？
        shard_index=jobid) # ？？？

    # 因为分布式的事情，指定一下设备
    # Note: One can set model_device_fn to
    # `tf.train.replica_device_setter(ps_tasks)` for distributed training.
    model_device_fn = None
    if extra_args: model_device_fn = extra_args.model_device_fn

    with tf.device(model_device_fn):
      model = model_creator( # 创建模型
          hparams,
          iterator=iterator,
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          source_vocab_table=src_vocab_table,
          target_vocab_table=tgt_vocab_table,
          scope=scope,
          extra_args=extra_args)

  # 返回一个“字典“，里面有图、模型、迭代器
  return TrainModel(
      graph=graph,
      model=model,
      iterator=iterator,
      skip_count_placeholder=skip_count_placeholder)


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "src_file_placeholder",
                            "tgt_file_placeholder", "iterator"))):
  pass


def create_eval_model(model_creator, hparams, scope=None, extra_args=None):
  """Create train graph, model, src/tgt file holders, and iterator."""
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file
  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "eval"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    # 数据文件都是占位？？？
    src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    src_dataset = tf.data.TextLineDataset(src_file_placeholder)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len_infer,
        tgt_max_len=hparams.tgt_max_len_infer)
    # 创建evaluation模型，
    # 问题： 训练模型和验证模型貌似不是一个图，怎么联系起来的呢？？？
    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.EVAL,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        scope=scope,
        extra_args=extra_args)
  return EvalModel(
      graph=graph,
      model=model,
      src_file_placeholder=src_file_placeholder,
      tgt_file_placeholder=tgt_file_placeholder,
      iterator=iterator)


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "src_placeholder",
                            "batch_size_placeholder", "iterator"))):
  pass


def create_infer_model(model_creator, hparams, scope=None, extra_args=None):
  """Create inference model."""
  graph = tf.Graph()
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  with graph.as_default(), tf.container(scope or "infer"):
    # 除了word2id，还有target的 id2word
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)

    # 数据占位，batch size 占位
    src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

    src_dataset = tf.data.Dataset.from_tensor_slices(
        src_placeholder)
    iterator = iterator_utils.get_infer_iterator(
        src_dataset,
        src_vocab_table,
        batch_size=batch_size_placeholder,
        eos=hparams.eos,
        src_max_len=hparams.src_max_len_infer)
    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.INFER,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        reverse_target_vocab_table=reverse_tgt_vocab_table,
        scope=scope,
        extra_args=extra_args)
  return InferModel(
      graph=graph,
      model=model,
      src_placeholder=src_placeholder,
      batch_size_placeholder=batch_size_placeholder,
      iterator=iterator)


def _get_embed_device(vocab_size): # ？？？分布式的东西？
  """Decide on which device to place an embed matrix given its vocab size."""
  if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
    return "/cpu:0"
  else:
    return "/gpu:0"


def _create_pretrained_emb_from_txt( # 加载一个预训练的embedding
    vocab_file, embed_file, num_trainable_tokens=3, dtype=tf.float32,
    scope=None):
  """Load pretrain embeding from embed_file, and return an embedding matrix.

  Args:
    embed_file: Path to a Glove formated embedding txt file.
    num_trainable_tokens: Make the first n tokens in the vocab file as trainable # 可训练的embedding的数量，前这么多个，默认有3个可训练
      variables. Default is 3, which is "<unk>", "<s>" and "</s>".
  """
  vocab, _ = vocab_utils.load_vocab(vocab_file)   # 加载词表
  trainable_tokens = vocab[:num_trainable_tokens] # 可训练的这些词

  utils.print_out("# Using pretrained embedding: %s." % embed_file)
  utils.print_out("  with trainable tokens: ")

  emb_dict, emb_size = vocab_utils.load_embed_txt(embed_file) # 从文本文件中加载预训练的embedding
  for token in trainable_tokens: # 遍历【可训练的这些词】，如果预训练的embedding中没有，则将其初始化为全零向量
    utils.print_out("    %s" % token)
    if token not in emb_dict:
      emb_dict[token] = [0.0] * emb_size

  emb_mat = np.array(
      [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
  emb_mat = tf.constant(emb_mat)
  # 将embedding矩阵中常数部分截取出来，token维度截取的范围是[num_trainable_tokens：-1]，embedding截取的维度是整个embedding[0：-1]
  emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1]) 
  # 将embedding矩阵中可训练部分抽取出来，貌似使用命名空间抽取的？？？
  with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype) as scope:
    with tf.device(_get_embed_device(num_trainable_tokens)):
      emb_mat_var = tf.get_variable(
          "emb_mat_var", [num_trainable_tokens, emb_size])
  return tf.concat([emb_mat_var, emb_mat_const], 0)   # 将常数embedding和可训练embedding拼接

# 生成一个embedding或者加载一个预训练好的embedding
def _create_or_load_embed(embed_name, vocab_file, embed_file, 
                          vocab_size, embed_size, dtype):
  """Create a new or load an existing embedding matrix."""
  if vocab_file and embed_file: # 从文本文件中加载embedding
    embedding = _create_pretrained_emb_from_txt(vocab_file, embed_file)
  else: # 生成没有经过预训练的embedding
    with tf.device(_get_embed_device(vocab_size)):
      embedding = tf.get_variable(
          embed_name, [vocab_size, embed_size], dtype)
  return embedding


def create_emb_for_encoder_and_decoder(share_vocab,   # encoder和decoder是否共用一个embedding
                                       src_vocab_size, # 词表大小
                                       tgt_vocab_size,
                                       src_embed_size, # embedding维度
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       num_partitions=0, # ？？？ 貌似是分布式训练的时候用到的
                                       src_vocab_file=None, # 词表文件
                                       tgt_vocab_file=None,
                                       src_embed_file=None, # embedding文件
                                       tgt_embed_file=None,
                                       scope=None):
  """Create embedding matrix for both encoder and decoder.

  Args:
    share_vocab: A boolean. Whether to share embedding matrix for both
      encoder and decoder.
    src_vocab_size: An integer. The source vocab size.
    tgt_vocab_size: An integer. The target vocab size.
    src_embed_size: An integer. The embedding dimension for the encoder's
      embedding.
    tgt_embed_size: An integer. The embedding dimension for the decoder's
      embedding.
    dtype: dtype of the embedding matrix. Default to float32.
    num_partitions: number of partitions used for the embedding vars.
    scope: VariableScope for the created subgraph. Default to "embedding".

  Returns:
    embedding_encoder: Encoder's embedding matrix.
    embedding_decoder: Decoder's embedding matrix.

  Raises:
    ValueError: if use share_vocab but source and target have different vocab
      size.
  """

  if num_partitions <= 1:
    partitioner = None
  else:
    # Note: num_partitions > 1 is required for distributed training due to
    # embedding_lookup tries to colocate single partition-ed embedding variable
    # with lookup ops. This may cause embedding variables being placed on worker
    # jobs.
    partitioner = tf.fixed_size_partitioner(num_partitions)

  if (src_embed_file or tgt_embed_file) and partitioner:
    raise ValueError(
        "Can't set num_partitions > 1 when using pretrained embedding")

  with tf.variable_scope(
      scope or "embeddings", dtype=dtype, partitioner=partitioner) as scope:
    # Share embedding
    if share_vocab: # 若encoder和decoder共用embedding
      if src_vocab_size != tgt_vocab_size:
        raise ValueError("Share embedding but different src/tgt vocab sizes"
                         " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
      assert src_embed_size == tgt_embed_size
      utils.print_out("# Use the same embedding for source and target")
      vocab_file = src_vocab_file or tgt_vocab_file
      embed_file = src_embed_file or tgt_embed_file

      embedding_encoder = _create_or_load_embed(
          "embedding_share", vocab_file, embed_file,
          src_vocab_size, src_embed_size, dtype)
      embedding_decoder = embedding_encoder
    else: # 不共用
      with tf.variable_scope("encoder", partitioner=partitioner):
        embedding_encoder = _create_or_load_embed(
            "embedding_encoder", src_vocab_file, src_embed_file,
            src_vocab_size, src_embed_size, dtype)

      with tf.variable_scope("decoder", partitioner=partitioner):
        embedding_decoder = _create_or_load_embed(
            "embedding_decoder", tgt_vocab_file, tgt_embed_file,
            tgt_vocab_size, tgt_embed_size, dtype)

  return embedding_encoder, embedding_decoder


def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, device_str=None, residual_fn=None):
  """创建一个rnn神经元
  参数：
    unit_type: 神经元种类，LSTM、GRU、NAS
    num_units: 内部维度？？？
    forget?_bias: ？？？
    dropout: 
    mode: 训练？验证？INFER？？？
    residual_connection: 是否有多余连接
    device_str: ？？？
    residual_fn: ？？？
  """
  """Create an instance of a single RNN cell."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0 # 仅在训练模式下进行dropout

  # Cell Type
  if unit_type == "lstm": # LSTM 单元
    utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
    single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units,
        forget_bias=forget_bias)
  elif unit_type == "gru": # GRU 单元
    utils.print_out("  GRU", new_line=False)
    single_cell = tf.contrib.rnn.GRUCell(num_units)
  elif unit_type == "layer_norm_lstm": # balabala 单元
    utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias,
                    new_line=False)
    single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units,
        forget_bias=forget_bias,
        layer_norm=True)
  elif unit_type == "nas": # NAS 单元
    utils.print_out("  NASCell", new_line=False)
    single_cell = tf.contrib.rnn.NASCell(num_units)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  # Dropout (= 1 - keep_prob)
  if dropout > 0.0: # 在之前生成的神经元基础上，添加dropout
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))
    utils.print_out("  %s, dropout=%g " %(type(single_cell).__name__, dropout),
                    new_line=False)

  # Residual
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)
    """
    residual_fn:
      The function to map raw cell inputs and raw cell outputs to the actual cell outputs of the residual network. 
      Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs and outputs.
      从文档的解释中可以看到，这个参数控制“多余连接”的方式；默认的方式是将神经元的输入和输出做“元求和”，作为最终的输出。
    """
    utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

  # Device Wrapper
  if device_str:
    single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
    utils.print_out("  %s, device=%s" %
                    (type(single_cell).__name__, device_str), new_line=False)

  return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0,
               single_cell_fn=None, residual_fn=None):
  """Create a list of RNN cells."""
  if not single_cell_fn:
    single_cell_fn = _single_cell

  # Multi-GPU
  cell_list = []
  for i in range(num_layers): 
    # 为什么是按照层数产生神经元数量呢？貌似是因为，这是一个time_step用到的网络结构；其他的神经元并不是真实的实例。
    utils.print_out("  cell %d" % i, new_line=False)
    single_cell = single_cell_fn(
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        dropout=dropout,
        mode=mode,
        residual_connection=(i >= num_layers - num_residual_layers), 
        # 当当前层的id大于等于(num_layers-num_residual_layers)时，将多余连接设置为true
        device_str=get_device_str(i + base_gpu, num_gpus),
        residual_fn=residual_fn
    )
    utils.print_out("")
    cell_list.append(single_cell)

  return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0,
                    single_cell_fn=None):
  """Create multi-layer RNN cell.

  Args:
    unit_type: string representing the unit type, i.e. "lstm".
    num_units: the depth of each unit.
    num_layers: number of cells.
    num_residual_layers: Number of residual layers from top to bottom. For
      example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
      cells in the returned list will be wrapped with `ResidualWrapper`.
    forget_bias: the initial forget bias of the RNNCell(s).
    dropout: floating point value between 0.0 and 1.0:
      the probability of dropout.  this is ignored if `mode != TRAIN`.
    mode: either tf.contrib.learn.TRAIN/EVAL/INFER
    num_gpus: The number of gpus to use when performing round-robin
      placement of layers.
    base_gpu: The gpu device id to use for the first RNN cell in the
      returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
      as its device id.
    single_cell_fn: allow for adding customized cell.
      When not specified, we default to model_helper._single_cell
  Returns:
    An `RNNCell` instance.
  """
  cell_list = _cell_list(unit_type=unit_type, # 创建一系列神经元，数量等同于网络层数
                         num_units=num_units,
                         num_layers=num_layers,
                         num_residual_layers=num_residual_layers,
                         forget_bias=forget_bias,
                         dropout=dropout,
                         mode=mode,
                         num_gpus=num_gpus,
                         base_gpu=base_gpu,
                         single_cell_fn=single_cell_fn)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list) # 这一步的意义在于将这些零散的神经元连起来？
    # 所以最终返回的是单步time step的多个神经元，个数是网络层数


def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

  return clipped_gradients, gradient_norm_summary, gradient_norm


def load_model(model, ckpt, session, name):
  start_time = time.time()
  model.saver.restore(session, ckpt)
  session.run(tf.tables_initializer())
  utils.print_out(
      "  loaded %s model parameters from %s, time %.2fs" %
      (name, ckpt, time.time() - start_time))
  return model


def avg_checkpoints(model_dir, num_last_checkpoints, global_step,
                    global_step_name):
  """Average the last N checkpoints in the model_dir."""
  checkpoint_state = tf.train.get_checkpoint_state(model_dir)
  if not checkpoint_state:
    utils.print_out("# No checkpoint file found in directory: %s" % model_dir)
    return None

  # 获取倒数N个checkpoints，如果不够则不进行平均
  # Checkpoints are ordered from oldest to newest.
  checkpoints = (
      checkpoint_state.all_model_checkpoint_paths[-num_last_checkpoints:])
  if len(checkpoints) < num_last_checkpoints:
    utils.print_out(
        "# Skipping averaging checkpoints because not enough checkpoints is "
        "avaliable."
    )
    return None

  # 准备好要存放平均模型的路径
  avg_model_dir = os.path.join(model_dir, "avg_checkpoints")
  if not tf.gfile.Exists(avg_model_dir):
    utils.print_out(
        "# Creating new directory %s for saving averaged checkpoints." %
        avg_model_dir)
    tf.gfile.MakeDirs(avg_model_dir)

  # 获取所有的变量，并进行全零初始化
  utils.print_out("# Reading and averaging variables in checkpoints:")
  var_list = tf.contrib.framework.list_variables(checkpoints[0])
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if name != global_step_name: # global_step_name是什么
      var_values[name] = np.zeros(shape)

  # 所有变量累加
  for checkpoint in checkpoints:
    utils.print_out("    %s" % checkpoint)
    reader = tf.contrib.framework.load_checkpoint(checkpoint)
    for name in var_values:
      tensor = reader.get_tensor(name)
      var_dtypes[name] = tensor.dtype
      var_values[name] += tensor

  for name in var_values: # 做平均
    var_values[name] /= len(checkpoints)

  # Build a graph with same variables in the checkpoints, and save the averaged
  # variables into the avg_model_dir.
  # 建个图，把平均变量值加载进去，然后把模型保存了
  with tf.Graph().as_default():
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
        for v in var_values
    ]

    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)] # 每个变量都创建一个赋值操作
    global_step_var = tf.Variable(
        global_step, name=global_step_name, trainable=False)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                             six.iteritems(var_values)):
        sess.run(assign_op, {p: value})

      # Use the built saver to save the averaged checkpoint. Only keep 1
      # checkpoint and the best checkpoint will be moved to avg_best_metric_dir.
      saver.save(
          sess,
          os.path.join(avg_model_dir, "translate.ckpt"))

  return avg_model_dir


def create_or_load_model(model, model_dir, session, name):
  """Create translation model and initialize or load parameters in session."""
  """先尝试加载最新模型，如果没有，则初始化模型，并告知用时
  """
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer()) # 全局变量初始化
    session.run(tf.tables_initializer()) # 初始化所有的表
    utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

  # tf.Variable().eval(): In a session, computes and returns the value of this variable.
  global_step = model.global_step.eval(session=session) 
  return model, global_step


def compute_perplexity(model, sess, name):
  """Compute perplexity of the output of the model.

  Args:
    model: model for compute perplexity.
    sess: tensorflow session to use.
    name: name of the batch.

  Returns:
    The perplexity of the eval outputs.
  """
  total_loss = 0
  total_predict_count = 0
  start_time = time.time()

  while True:
    try:
      loss, predict_count, batch_size = model.eval(sess)
      total_loss += loss * batch_size
      total_predict_count += predict_count
    except tf.errors.OutOfRangeError:
      break

  perplexity = utils.safe_exp(total_loss / total_predict_count)
  utils.print_time("  eval %s: perplexity %.2f" % (name, perplexity),
                   start_time)
  return perplexity
