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

"""TensorFlow NMT model implementation."""
from __future__ import print_function

import argparse
import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from . import inference
from . import train
from .utils import evaluation_utils
from .utils import misc_utils as utils
from .utils import vocab_utils

utils.check_tensorflow_version()

FLAGS = None


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # network
  parser.add_argument("--num_units", type=int, default=32, help="Network size.")
  parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
  parser.add_argument("--num_encoder_layers", type=int, default=None, help="Encoder depth, equal to num_layers if None.")
  parser.add_argument("--num_decoder_layers", type=int, default=None, help="Decoder depth, equal to num_layers if None.")
  parser.add_argument("--encoder_type", type=str, default="uni", help="""\
      uni | bi | gnmt.
      For bi, we build num_encoder_layers/2 bi-directional layers.
      For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1)
        uni-directional layers.\
      """)
  parser.add_argument("--residual", type="bool", nargs="?", const=True,
                      default=False,
                      help="Whether to add residual connections.")
  parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                      default=True,
                      help="Whether to use time-major mode for dynamic RNN.")
  # ？？？
  parser.add_argument("--num_embeddings_partitions", type=int, default=0,
                      help="Number of partitions for embedding vars.")

  # attention mechanisms
  parser.add_argument("--attention", type=str, default="", help="""\
      luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
      attention\
      """)
  parser.add_argument(
      "--attention_architecture",
      type=str,
      default="standard",
      help="""\
      standard | gnmt | gnmt_v2.
      standard: use top layer to compute attention.
      gnmt: GNMT style of computing attention, use previous bottom layer to
          compute attention.
      gnmt_v2: similar to gnmt, but use current bottom layer to compute
          attention.\
      """)
  parser.add_argument(
      "--output_attention", type="bool", nargs="?", const=True,
      default=True,
      help="""\
      Only used in standard attention_architecture. Whether use attention as
      the cell output at each timestep.
      .\
      """)
  parser.add_argument(
      "--pass_hidden_state", type="bool", nargs="?", const=True,
      default=True,
      help="""\
      Whether to pass encoder's hidden state to decoder when using an attention
      based model.\
      """)

  # optimizer
  parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
  parser.add_argument("--learning_rate", type=float, default=1.0,
                      help="Learning rate. Adam: 0.001 | 0.0001")
  # 以下3个参数都是控制 学习率的动态调整的，？？？
  parser.add_argument("--warmup_steps", type=int, default=0,
                      help="How many steps we inverse-decay learning.")
  parser.add_argument("--warmup_scheme", type=str, default="t2t", help="""\
      How to warmup learning rates. Options include:
        t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
             exponentiate until the specified lr.\
      """)
  parser.add_argument(
      "--decay_scheme", type=str, default="", help="""\
      How we decay learning rate. Options include:
        luong234: after 2/3 num train steps, we start halving the learning rate
          for 4 times before finishing.
        luong5: after 1/2 num train steps, we start halving the learning rate
          for 5 times before finishing.\
        luong10: after 1/2 num train steps, we start halving the learning rate
          for 10 times before finishing.\
      """)

  parser.add_argument(
      "--num_train_steps", type=int, default=12000, help="Num steps to train.")
  # ？？？ 防止梯度爆炸的东西？
  parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                      const=True,
                      default=True,
                      help=("Whether try colocating gradients with "
                            "corresponding op"))

  # initializer
  parser.add_argument("--init_op", type=str, default="uniform",
                      help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help=("for uniform init_op, initialize weights "
                            "between [-this, this]."))

  # data # 数据文件名、前缀即路径、模型和日志输出路径
  parser.add_argument("--src", type=str, default=None,
                      help="Source suffix, e.g., en.")
  parser.add_argument("--tgt", type=str, default=None,
                      help="Target suffix, e.g., de.")
  parser.add_argument("--train_prefix", type=str, default=None,
                      help="Train prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--dev_prefix", type=str, default=None,
                      help="Dev prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--test_prefix", type=str, default=None,
                      help="Test prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--out_dir", type=str, default=None,
                      help="Store log/model files.")

  # Vocab
  parser.add_argument("--vocab_prefix", type=str, default=None, help="""\
      Vocab prefix, expect files with src/tgt suffixes.\
      """)
  parser.add_argument("--embed_prefix", type=str, default=None, help="""\
      Pretrained embedding prefix, expect files with src/tgt suffixes.
      The embedding files should be Glove formated txt files.\
      """)
  parser.add_argument("--sos", type=str, default="<s>",
                      help="Start-of-sentence symbol.")
  parser.add_argument("--eos", type=str, default="</s>",
                      help="End-of-sentence symbol.")
  parser.add_argument("--share_vocab", type="bool", nargs="?", const=True,
                      default=False,
                      help="""\
      Whether to use the source vocab and embeddings for both source and
      target.\
      """)
  parser.add_argument("--check_special_token", type="bool", default=True,
                      help="""\
                      Whether check special sos, eos, unk tokens exist in the
                      vocab files.\
                      """)

  # Sequence lengths # 句子序列最大长度限制
  parser.add_argument("--src_max_len", type=int, default=50,
                      help="Max length of src sequences during training.")
  parser.add_argument("--tgt_max_len", type=int, default=50,
                      help="Max length of tgt sequences during training.")
  parser.add_argument("--src_max_len_infer", type=int, default=None,
                      help="Max length of src sequences during inference.")
  parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                      help="""\
      Max length of tgt sequences during inference.  Also use to restrict the
      maximum decoding length.\
      """)

  # Default settings works well (rarely need to change)
  parser.add_argument("--unit_type", type=str, default="lstm",
                      help="lstm | gru | layer_norm_lstm | nas")
  # ？？？
  parser.add_argument("--forget_bias", type=float, default=1.0,
                      help="Forget bias for BasicLSTMCell.")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (not keep_prob)")
  # ？？？
  parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
  parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

  parser.add_argument("--steps_per_stats", type=int, default=100, # 做一次累计的间隔步数
                      help=("How many training steps to do per stats logging."
                            "Save checkpoint every 10x steps_per_stats"))
  parser.add_argument("--max_train", type=int, default=0,
                      help="Limit on the size of training data (0: no limit).")
  # 将数据按照句长分组，用处是什么？？？
  parser.add_argument("--num_buckets", type=int, default=5,
                      help="Put data into similar-length buckets.")

  # SPM # 还不知道怎么用的？？？
  parser.add_argument("--subword_option", type=str, default="",
                      choices=["", "bpe", "spm"],
                      help="""\
                      Set to bpe or spm to activate subword desegmentation.\
                      """)

  # Misc
  parser.add_argument("--num_gpus", type=int, default=1,
                      help="Number of gpus in each worker.")
  # ？？？
  parser.add_argument("--log_device_placement", type="bool", nargs="?",
                      const=True, default=False, help="Debug GPU allocation.")
  parser.add_argument("--metrics", type=str, default="bleu", # 评价指标
                      help=("Comma-separated list of evaluations "
                            "metrics (bleu,rouge,accuracy)"))
  # external evaluation 应该是指平均模型的验证
  parser.add_argument("--steps_per_external_eval", type=int, default=None, 
                      help="""\
      How many training steps to do per external evaluation.  Automatically set
      based on data if None.\
      """)
  parser.add_argument("--scope", type=str, default=None, # 全局scope吗，
                      help="scope to put variables under")
  # 这是标准参数的路径，然而并不知道标准参数、FLAGS这些参数的区别
  parser.add_argument("--hparams_path", type=str, default=None, 
                      help=("Path to standard hparams json file that overrides"
                            "hparams values from FLAGS."))
  parser.add_argument("--random_seed", type=int, default=None,
                      help="Random seed (>0, set a specific seed).")
  # 谁复写谁？？？
  parser.add_argument("--override_loaded_hparams", type="bool", nargs="?",
                      const=True, default=False,
                      help="Override loaded hparams with values specified")
  parser.add_argument("--num_keep_ckpts", type=int, default=5,
                      help="Max number of checkpoints to keep.")
  parser.add_argument("--avg_ckpts", type="bool", nargs="?",
                      const=True, default=False, help=("""\
                      Average the last N checkpoints for external evaluation.
                      N can be controlled by setting --num_keep_ckpts.\
                      """))

  # Inference
  # 加载哪个checkpoint来进行测试
  parser.add_argument("--ckpt", type=str, default="",
                      help="Checkpoint file to load a model for inference.")
  # 被翻译的语料文件
  parser.add_argument("--inference_input_file", type=str, default=None,
                      help="Set to the text to decode.")
  # ？？？ 
  # 字面意思：逗号分隔的句子指数
  parser.add_argument("--inference_list", type=str, default=None,
                      help=("A comma-separated list of sentence indices "
                            "(0-based) to decode."))
  parser.add_argument("--infer_batch_size", type=int, default=32,
                      help="Batch size for inference mode.")
  parser.add_argument("--inference_output_file", type=str, default=None,
                      help="Output file to store decoding results.")
  # 用来计算评价分数的引用文件？？？
  parser.add_argument("--inference_ref_file", type=str, default=None,
                      help=("""\
      Reference file to compute evaluation scores (if provided).\
      """))
  parser.add_argument("--beam_width", type=int, default=0,
                      help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
  # 用在beam search中来惩罚句子长度的参数，原理还不知道
  parser.add_argument("--length_penalty_weight", type=float, default=0.0,
                      help="Length penalty for beam search.")
  parser.add_argument("--sampling_temperature", type=float, # 同上
                      default=0.0,
                      help=("""\
      Softmax sampling temperature for inference decoding, 0.0 means greedy
      decoding. This option is ignored when using beam search.\
      """))
  parser.add_argument("--num_translations_per_input", type=int, default=1,
                      help=("""\
      Number of translations generated for each sentence. This is only used for
      inference.\
      """))

  # Job info # ？？？
  parser.add_argument("--jobid", type=int, default=0,
                      help="Task id of the worker.")
  parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of workers (inference only).")
  parser.add_argument("--num_inter_threads", type=int, default=0,
                      help="number of inter_op_parallelism_threads")
  parser.add_argument("--num_intra_threads", type=int, default=0,
                      help="number of intra_op_parallelism_threads")


def create_hparams(flags):
  """Create training hparams."""
  # 函数功能： 将 argparse.Namespace 转换为 tf.contrib.training.HParams 实例
  #           因为后者还有前者不具备的功能，例如添加参数
  # Arg：
  #   flags: 是一个argparse.Namespace实例
  return tf.contrib.training.HParams(
      # Data
      src=flags.src,
      tgt=flags.tgt,
      train_prefix=flags.train_prefix,
      dev_prefix=flags.dev_prefix,
      test_prefix=flags.test_prefix,
      vocab_prefix=flags.vocab_prefix,
      embed_prefix=flags.embed_prefix,
      out_dir=flags.out_dir,

      # Networks
      num_units=flags.num_units, # 神经元向量维度
      num_layers=flags.num_layers,  # Compatible # 网络层数
      num_encoder_layers=(flags.num_encoder_layers or flags.num_layers), # encoder网络层数
      num_decoder_layers=(flags.num_decoder_layers or flags.num_layers), # decoder网络层数
      dropout=flags.dropout, # 
      unit_type=flags.unit_type, # LSTM/GRU/NAS
      encoder_type=flags.encoder_type, # ？？？
      residual=flags.residual, # 是否使用额外连接？？？
      time_major=flags.time_major, #
      num_embeddings_partitions=flags.num_embeddings_partitions, # ？？？

      # Attention mechanisms
      attention=flags.attention,
      attention_architecture=flags.attention_architecture,
      output_attention=flags.output_attention,
      pass_hidden_state=flags.pass_hidden_state,

      # Train 
      optimizer=flags.optimizer, # 优化方式 GSD/ADAM
      num_train_steps=flags.num_train_steps, # ？？？
      batch_size=flags.batch_size, # 
      init_op=flags.init_op, # 初始化方式 正态/均匀
      init_weight=flags.init_weight, # 初始化范围
      max_gradient_norm=flags.max_gradient_norm, # ？？？
      learning_rate=flags.learning_rate, #
      warmup_steps=flags.warmup_steps, # ？？？
      warmup_scheme=flags.warmup_scheme, # ？？？
      decay_scheme=flags.decay_scheme, # ？？？
      colocate_gradients_with_ops=flags.colocate_gradients_with_ops, # ？？？

      # Data constraints
      num_buckets=flags.num_buckets, # ？？？
      max_train=flags.max_train, # ？？？
      src_max_len=flags.src_max_len, # 最大句长
      tgt_max_len=flags.tgt_max_len, # 最大句长

      # Inference
      src_max_len_infer=flags.src_max_len_infer,
      tgt_max_len_infer=flags.tgt_max_len_infer,
      infer_batch_size=flags.infer_batch_size, #
      beam_width=flags.beam_width, # 
      length_penalty_weight=flags.length_penalty_weight, # 具体惩罚原理不清楚？？？
      sampling_temperature=flags.sampling_temperature,
      num_translations_per_input=flags.num_translations_per_input,

      # Vocab
      sos=flags.sos if flags.sos else vocab_utils.SOS,
      eos=flags.eos if flags.eos else vocab_utils.EOS,
      subword_option=flags.subword_option,
      check_special_token=flags.check_special_token,

      # Misc
      forget_bias=flags.forget_bias,
      num_gpus=flags.num_gpus,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=flags.steps_per_stats,
      steps_per_external_eval=flags.steps_per_external_eval,
      share_vocab=flags.share_vocab,
      metrics=flags.metrics.split(","),
      log_device_placement=flags.log_device_placement,
      random_seed=flags.random_seed,
      override_loaded_hparams=flags.override_loaded_hparams,
      num_keep_ckpts=flags.num_keep_ckpts,
      avg_ckpts=flags.avg_ckpts,
      num_intra_threads=flags.num_intra_threads,
      num_inter_threads=flags.num_inter_threads,
  )


def extend_hparams(hparams):
  """Extend training hparams."""
  # 函数功能： 参数预处理，根据现有参数生成一些新的参数，
  #           例如将一些目录添加到参数列表，检查创建这些目录

  # encoder 和 decoder 层数不同，则中间的隐藏状态传不过去
  assert hparams.num_encoder_layers and hparams.num_decoder_layers
  if hparams.num_encoder_layers != hparams.num_decoder_layers:
    hparams.pass_hidden_state = False
    utils.print_out("Num encoder layer %d is different from num decoder layer"
                    " %d, so set pass_hidden_state to False" % (
                        hparams.num_encoder_layers,
                        hparams.num_decoder_layers))

  # Sanity checks
  # 如果encoder是双向的，那么层数必须是偶数，这里做个检查
  if hparams.encoder_type == "bi" and hparams.num_encoder_layers % 2 != 0:
    raise ValueError("For bi, num_encoder_layers %d should be even" %
                     hparams.num_encoder_layers)
  if (hparams.attention_architecture in ["gnmt"] and
      hparams.num_encoder_layers < 2):
    raise ValueError("For gnmt attention architecture, "
                     "num_encoder_layers %d should be >= 2" %
                     hparams.num_encoder_layers)

  # Set residual layers
  # 关于额外连接的参数预运算
  num_encoder_residual_layers = 0
  num_decoder_residual_layers = 0
  if hparams.residual: # 额外连接的层数要比总层数少1
    if hparams.num_encoder_layers > 1:
      num_encoder_residual_layers = hparams.num_encoder_layers - 1
    if hparams.num_decoder_layers > 1:
      num_decoder_residual_layers = hparams.num_decoder_layers - 1

    # 如果是gnmt网络，那么由于前两层是双向网络，所以额外连接要再减1
    if hparams.encoder_type == "gnmt": 
      # The first unidirectional layer (after the bi-directional layer) in
      # the GNMT encoder can't have residual connection due to the input is
      # the concatenation of fw_cell and bw_cell's outputs.
      num_encoder_residual_layers = hparams.num_encoder_layers - 2

      # Compatible for GNMT models
      if hparams.num_encoder_layers == hparams.num_decoder_layers:
        num_decoder_residual_layers = num_encoder_residual_layers
  # 算出额外连接的层数之后，放到参数里面
  hparams.add_hparam("num_encoder_residual_layers", num_encoder_residual_layers)
  hparams.add_hparam("num_decoder_residual_layers", num_decoder_residual_layers)

  # subword机制的参数，目前还不知道？？？
  if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
    raise ValueError("subword option must be either spm, or bpe")

  # Flags
  utils.print_out("# hparams:")
  utils.print_out("  src=%s" % hparams.src)
  utils.print_out("  tgt=%s" % hparams.tgt)
  utils.print_out("  train_prefix=%s" % hparams.train_prefix)
  utils.print_out("  dev_prefix=%s" % hparams.dev_prefix)
  utils.print_out("  test_prefix=%s" % hparams.test_prefix)
  utils.print_out("  out_dir=%s" % hparams.out_dir)

  ## Vocab
  # Get vocab file names first
  # 获得词表文件，之后将其文件路径（目录+文件名）放到参数列表中
  if hparams.vocab_prefix:
    src_vocab_file = hparams.vocab_prefix + "." + hparams.src
    tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
  else:
    raise ValueError("hparams.vocab_prefix must be provided.")

  # Source vocab
  src_vocab_size, src_vocab_file = vocab_utils.check_vocab( # 检查处理词表，不存在则报错
      src_vocab_file, # 词表文件名
      hparams.out_dir, # 词表存放的目录
      check_special_token=hparams.check_special_token, # 是否检查helpwords存在与否
      sos=hparams.sos, # 3个help words
      eos=hparams.eos,
      unk=vocab_utils.UNK)

  # Target vocab
  if hparams.share_vocab: # 共享词表
    utils.print_out("  using source vocab for target")
    tgt_vocab_file = src_vocab_file
    tgt_vocab_size = src_vocab_size
  else: # target自己的词表
    tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
        tgt_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)
  hparams.add_hparam("src_vocab_size", src_vocab_size)
  hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
  hparams.add_hparam("src_vocab_file", src_vocab_file)
  hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

  # Pretrained Embeddings:
  # 获取设置 embedding 文件参数
  hparams.add_hparam("src_embed_file", "")
  hparams.add_hparam("tgt_embed_file", "")
  if hparams.embed_prefix:
    src_embed_file = hparams.embed_prefix + "." + hparams.src
    tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt

    if tf.gfile.Exists(src_embed_file):
      hparams.src_embed_file = src_embed_file

    if tf.gfile.Exists(tgt_embed_file):
      hparams.tgt_embed_file = tgt_embed_file

  # Check out_dir
  # 看来这个输出目录包括各种IO文件，此处检查目录是否存在，不存在则创建
  if not tf.gfile.Exists(hparams.out_dir):
    utils.print_out("# Creating output directory %s ..." % hparams.out_dir)
    tf.gfile.MakeDirs(hparams.out_dir)

  # Evaluation
  # metrics是评价指标，有(bleu,rouge,accuracy)
  for metric in hparams.metrics: 
    # 这意思是可以同时计算多个评价指标，并且为了记录每个评价指标的最优值，为每个建了一个文件夹，保存最优值
    hparams.add_hparam("best_" + metric, 0)  # larger is better
    best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
    hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
    tf.gfile.MakeDirs(best_metric_dir)

    if hparams.avg_ckpts: # 如果要测试平均模型，还为每个评价指标建了个文件夹，文件名中带着avg
      hparams.add_hparam("avg_best_" + metric, 0)  # larger is better
      best_metric_dir = os.path.join(hparams.out_dir, "avg_best_" + metric)
      hparams.add_hparam("avg_best_" + metric + "_dir", best_metric_dir)
      tf.gfile.MakeDirs(best_metric_dir)

  return hparams


def ensure_compatible_hparams(hparams, default_hparams, hparams_path):
  """Make sure the loaded hparams is compatible with new changes."""
  # hparams： 从文件加载的参数
  # default_hparams： 解析获得的参数
  # hparams_path： 标准参数

  # 从下面的逻辑，看出3种参数的关系，真TM乱
  # 首先，如果用标准参数，那么 标准参数 > 解析参数
  # 其次，可以由用户指定，是否使用 解析参数 覆盖 加载的参数

  # 可以简单理解为： 标准参数 > 解析参数 > 外部加载的参数

  # 首先从hparams_path加载standard params，覆盖对应的 default params
  default_hparams = utils.maybe_parse_standard_hparams(default_hparams, hparams_path)

  # For compatible reason, if there are new fields in default_hparams,
  #   we add them to the current hparams
  # 合并 default params 和 hparams，优先级： hparams > default_hparams
  default_config = default_hparams.values()
  config = hparams.values()
  for key in default_config:
    if key not in config:
      hparams.add_hparam(key, default_config[key])

  # Update all hparams' keys if override_loaded_hparams=True
  # 如果可以，用新解析的参数覆盖从文件中加载的参数
  if default_hparams.override_loaded_hparams:
    for key in default_config:
      if getattr(hparams, key) != default_config[key]:
        utils.print_out("# Updating hparams.%s: %s -> %s" %
                        (key, str(getattr(hparams, key)),
                         str(default_config[key])))
        setattr(hparams, key, default_config[key])
  return hparams


def create_or_load_hparams(
    out_dir, default_hparams, hparams_path, save_hparams=True):
  """Create hparams or load hparams from out_dir."""
  # 函数功能：
  #   总结来说，涉及到3种参数，解析参数default_hparams，标准参数hparams_path，存储参数out_dir
  #   3种参数的优先级可以简单理解为：  标准参数 > 解析参数 > 外部加载的参数
  # Args:
  #   out_dir: 外部参数加载、保存路径
  #   default_hparams： tf.contrib.training.HParams类实例，解析获得的参数
  #   hparams_path： 标准参数的路径
  # Return:
  #   合并后的参数，tf.contrib.training.HParams类实例

  # 从外部文件加载参数
  # 获得tf.contrib.training.HParams类实例或者None
  hparams = utils.load_hparams(out_dir) 
  
  if not hparams: 
    # 如果外部参数不存在： 
    #   用standard params 覆盖 default params
    #   扩展参数，即进行一些参数预处理（例如知道了网络层数，提前计算一下额外连接层数）
    hparams = default_hparams 
    hparams = utils.maybe_parse_standard_hparams(hparams, hparams_path) # 从hparams_path加载新参数，覆盖hparams
    hparams = extend_hparams(hparams)
  else: 
    # 如果外部参数存在，则将外部参数、解析参数、标准参数合并，还是想说真TM乱
    hparams = ensure_compatible_hparams(hparams, default_hparams, hparams_path)

  # Save HParams
  if save_hparams:
    utils.save_hparams(out_dir, hparams)
    for metric in hparams.metrics:
      utils.save_hparams(getattr(hparams, "best_" + metric + "_dir"), hparams)

  # Print HParams
  utils.print_hparams(hparams)
  return hparams


def run_main(flags, default_hparams, train_fn, inference_fn, target_session=""):
  """Run main."""
  """
    flags: argparse.Namespace类实例,是解析成功的参数
    default_hparams: tf.contrib.training.HParams类实例
    train_fn: train function
    inference_fn: inference function
  """
  # Job
  jobid = flags.jobid # ？？？ jobid是什么
  num_workers = flags.num_workers
  utils.print_out("# Job id %d" % jobid)

  # Random
  # 设置random和numpy的随机数种子
  random_seed = flags.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed + jobid)
    np.random.seed(random_seed + jobid)

  ## Train / Decode
  out_dir = flags.out_dir # 从下面create_or_load_hparams函数来看，这个路径存储参数，模型是否也存储在此？？？
  if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

  # Load hparams.
  # default_hparams 中是程序解析获得的参数，
  # 下面这个函数，从解析参数、存储参数、标准参数3部分中获取参数，并处理兼容，形成最终的参数
  # 简单地理解，标准参数 > 解析参数 > 外部加载的参数
  hparams = create_or_load_hparams(
      out_dir, default_hparams, flags.hparams_path, save_hparams=(jobid == 0))

  if flags.inference_input_file: 
    # 如果设置了inference语料，则进行翻译，否则进行训练

    # Inference indices
    hparams.inference_indices = None 
    if flags.inference_list:
      (hparams.inference_indices) = ( # 也就是说，inference_indices是一个列表，元素为句子的id，这些句子是等待翻译的句子
          [int(token)  for token in flags.inference_list.split(",")])

    # Inference
    trans_file = flags.inference_output_file # 翻译之后输出到什么文件
    ckpt = flags.ckpt # flag.ckpt应该是指定使用哪个模型来进行翻译，如果不指定就使用最新的模型进行翻译
    if not ckpt:
      ckpt = tf.train.latest_checkpoint(out_dir)
    inference_fn(ckpt, flags.inference_input_file,
                 trans_file, hparams, num_workers, jobid)
    # 进行翻译用的参数
    # ckpt： 模型
    # flags.inference_input_file： 输入文件，内容应该是分行的，每行一个待翻译的句子
    # trans_file: 翻译输出文件
    # hparams： 至少有一个hparams.inference_indices参数，里面是句子idices

    # Evaluation
    ref_file = flags.inference_ref_file
    if ref_file and tf.gfile.Exists(trans_file):
      for metric in hparams.metrics:
        score = evaluation_utils.evaluate(
            ref_file,
            trans_file,
            metric,
            hparams.subword_option)
        utils.print_out("  %s: %.1f" % (metric, score))
  else:
    # 如果没有设置inference语料，则进行模型训练，否则进行翻译
    # Train
    train_fn(hparams, target_session=target_session)


def main(unused_argv):
  # Arg：
  #   unused_argv： 代码本身文件路径和解析器不认识的参数

  # FLAGS 是解析成功的参数，是一个argparse.Namespace类实例
  # 返回一个tf.contrib.training.HParams类实例
  # 这一步功能： 将参数放到tf.contrib.training.HParams实例中，方便之后添加参数
  default_hparams = create_hparams(FLAGS) 
  
  train_fn = train.train # 从train.py中引用的train()函数
  inference_fn = inference.inference# inference.py中引用的inference()函数
  run_main(FLAGS, default_hparams, train_fn, inference_fn)
  # FLAGS 和 default_hparams 什么关系？感觉重复了？？？


if __name__ == "__main__":
  # 构建参数解析器，添加参数项
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)

  # 
  FLAGS, unparsed = nmt_parser.parse_known_args()
  # FLAGS是解析成功的参数，
  #   是一个argparse.Namespace类实例，用FLAGS.param可以访问参数
  #   解析成功的参数包括命中的参数和具有默认值的参数
  # unparsed是不认识的没有解析的参数，key和value被一视同仁作为字符串保存在一个列表中
  # 例子见 https://blog.csdn.net/m0_37041325/article/details/77934623

  # 运行
  # FLAGS是全局变量，所以没有传进run函数
  # sys.argv[0]表示代码本身文件路径
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
