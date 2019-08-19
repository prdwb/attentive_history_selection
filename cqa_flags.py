from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import modeling
import optimization
import tokenization
import six
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# for running in jupyter env
flags.DEFINE_string('f', '', 'kernel')

# Required parameters

# BERT-base

flags.DEFINE_string(
    "bert_config_file", "/mnt/scratch/chenqu/bert/uncased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "/mnt/scratch/chenqu/bert/uncased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", "/mnt/scratch/chenqu/bert/uncased_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "output_dir", "/mnt/scratch/chenqu/bert_out/100000/",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("coqa_train_file", "/mnt/scratch/chenqu/coqa_extractive_gt/coqa-train-v1.0.json",
                    "CoQA json for training. E.g., coqa-train-v1.0.json")

flags.DEFINE_string(
    "coqa_predict_file", "/mnt/scratch/chenqu/coqa_extractive_gt/coqa-dev-v1.0.json",
    "CoQA json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string("quac_train_file", "/mnt/scratch/chenqu/quac_original/train_v0.2.json",
                    "QuAC json for training.")

flags.DEFINE_string(
    "quac_predict_file", "/mnt/scratch/chenqu/quac_original/val_v0.2.json",
    "QuAC json for predictions.")


flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384, # 384
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 12, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 12,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 2.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.0,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("evaluation_steps", 5,
                     "How often to do evaluation.")

flags.DEFINE_integer("evaluate_after", 4,
                     "we do evaluation after centain steps.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 40,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_integer(
    "history", 6,
    "Number of conversation history to use. applicable to the 'previous_j rule'"
)


flags.DEFINE_bool(
    "only_history_answer", True,
    "only prepend history answers without questions?")

flags.DEFINE_bool(
    "use_history_answer_marker", True,
    "use markers for hisotory answers instead of prepending them."
    "This flag surpasses the only_history_answer flag.")

flags.DEFINE_bool(
    "load_small_portion", True,
    "during develping, we only want to load a very small portion of "
    "the data to see if the code works.")

# no longer used
flags.DEFINE_bool(
    "use_RL", False,
    "whether to use the reinforced backtracker."
    "this flag supasses the history flag, because we will choose history freely with RL")

flags.DEFINE_string(
    "dataset", 'quac',
    "QuAC")

# only used in history attention
flags.DEFINE_integer(
    "max_history_turns", 11,
    "what is the max history turns a question can have "
    "e.g. in QuAC data, a dialog has a maximum of 12 turns,"
    "so a question has a maximum of 11 history turns"
    "however, if FLAGS.append_self is True, we need to set it to 12 "
    "because the variation without history is considered as a 'turn rep'") 

# no longer used
flags.DEFINE_integer("example_batch_size", 4, "when using RL, we want the batch size to be smaller because one example can gen multiple features")

flags.DEFINE_string(
    "cache_dir", "/mnt/scratch/chenqu/test_ham_cache/",
    "we store generated features here, so that we do not need to generate them every time")

flags.DEFINE_integer(
    "max_considered_history_turns", 11,
    "we only consider k history turns that immediately proceed the current turn, when generating preprocessed features,"
    "training will be slow if this is set to a large number")

flags.DEFINE_integer(
    "train_steps", 20,
    "loss: the loss gap on reward set, f1: the f1 on reward set")

flags.DEFINE_bool(
    "better_hae", True,
    "assign different history answer embedding to differet previous turns (PosHAE)")

flags.DEFINE_string(
    "history_selection", "previous_j", ""
)

flags.DEFINE_integer(
    "more_history", 0,
    "Number of conversation history to use. applicable to other rules except for previous_j")

flags.DEFINE_integer(
    "max_question_len_for_matching", 20,
    "applicable for the interaction matrix (tokens)")

flags.DEFINE_integer(
    "max_answer_len_for_matching", 40,
    "applicable for the interaction matrix (tokens)")

flags.DEFINE_string(
    "glove", '/mnt/scratch/chenqu/glove/glove.840B.300d.pkl',
    "glove pre-trained word embedding, we use 840B.300d")

flags.DEFINE_integer(
    "embedding_dim", 300,
    "dimension for glove pre-trained word embedding")

flags.DEFINE_integer(
    "kernel_size", 3,
    "cnn kernel size for the cnn in policy net")

flags.DEFINE_integer(
    "kernel_count", 16,
    "cnn kernel count for the cnn in policy net")

flags.DEFINE_integer(
    "pool_size", 3,
    "cnn kernel size for the cnn in policy net")

flags.DEFINE_float("rl_learning_rate", 1e-4, "The initial learning rate for the policy net and value net.")

flags.DEFINE_bool("MTL", True, "multi-task learning. jointly learn the dialog acts (followup, yesno)")

flags.DEFINE_float("MTL_lambda", 0.1, "total loss = (1 - 2 * lambda) * convqa_loss + lambda * followup_loss + lambda * yesno_loss")

flags.DEFINE_float("MTL_mu", 0.8, "total loss = mu * convqa_loss + lambda * followup_loss + lambda * yesno_loss")

flags.DEFINE_integer(
    "ideal_selected_num", 1,
    "ideal # selected history turns per example/question")

flags.DEFINE_bool("aux", False, "use aux loss or not")

flags.DEFINE_float("aux_lambda", 0.0, "auxiliary loss")

flags.DEFINE_bool("aux_shared", False, "wheter to share the aux prediction layer with the main convqa model")

flags.DEFINE_bool("disable_attention", False, "dialable the history attention module")

flags.DEFINE_bool("history_attention_hidden", False, "include hidden layers for the history att module")

flags.DEFINE_string("history_attention_input", "reduce_mean", "CLS, reduce_mean, reduce_max")

flags.DEFINE_string("mtl_input", "reduce_mean", "CLS, reduce_mean, reduce_max")

flags.DEFINE_integer("history_ngram", 1, 
               "in history attention, we attend to groups of history turns, this param indicate how many histories in one group"
               "if set to 1, it's equivalent to attend to every history turns independently"     )

flags.DEFINE_bool("reformulate_question", False, "prepend the immediate previous history question to the current question")

flags.DEFINE_bool("front_padding", False, "pad the BERT input sequence at the front")

flags.DEFINE_bool("freeze_bert", False, "freeze BERT")

flags.DEFINE_bool("fine_grained_attention", True, "use fine grained attention")

flags.DEFINE_bool("append_self", False, "when converting an example to variations, whether to append a variation without any history (self)")

flags.DEFINE_float("null_score_diff_threshold", 0.0, "null_score_diff_threshold")

flags.DEFINE_integer("bert_hidden", 768, "bert hidden units, 768 or 1024")