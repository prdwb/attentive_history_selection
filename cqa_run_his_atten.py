
# coding: utf-8

# In[1]:


# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[2]:


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
import numpy as np
from copy import deepcopy
import pickle
import itertools
from time import time
import traceback

from cqa_supports import *
from cqa_flags import FLAGS
from cqa_model import *
from cqa_gen_batches import *
# from cqa_rl_supports import *
from scorer import external_call # quac official evaluation script


# In[3]:


# for key in FLAGS:
#     print(key, ':', FLAGS[key].value)
print('Arguments:')
for key in FLAGS.__flags.keys():
    print('  {}: {}'.format(key, getattr(FLAGS, key)))
print('====')

tf.set_random_seed(0)
tf.logging.set_verbosity(tf.logging.INFO)
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

tf.gfile.MakeDirs(FLAGS.output_dir)
tf.gfile.MakeDirs(FLAGS.output_dir + '/summaries/train/')
tf.gfile.MakeDirs(FLAGS.output_dir + '/summaries/val/')
tf.gfile.MakeDirs(FLAGS.output_dir + '/summaries/rl/')

tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

print('output_dir', FLAGS.output_dir)

if FLAGS.append_self:
    FLAGS.cache_dir = FLAGS.cache_dir[:-1] + '_append_self/'
    
if FLAGS.max_seq_length != 384:
    FLAGS.cache_dir = FLAGS.cache_dir[:-1] + '_{}/'.format(FLAGS.max_seq_length)

if FLAGS.do_train:
    # read in training data, generate training features, and generate training batches
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    train_file = FLAGS.quac_train_file
    train_examples = read_quac_examples(input_file=train_file, is_training=True)
        
    
    # we attempt to read features from cache
    features_fname = FLAGS.cache_dir + FLAGS.dataset.lower() +                                    '/train_features_{}_{}.pkl'.format(FLAGS.load_small_portion, FLAGS.max_considered_history_turns)
    example_tracker_fname = FLAGS.cache_dir + FLAGS.dataset.lower() +                                    '/example_tracker_{}_{}.pkl'.format(FLAGS.load_small_portion, FLAGS.max_considered_history_turns)
    variation_tracker_fname = FLAGS.cache_dir + FLAGS.dataset.lower() +                                    '/variation_tracker_{}_{}.pkl'.format(FLAGS.load_small_portion, FLAGS.max_considered_history_turns)
    example_features_nums_fname = FLAGS.cache_dir + FLAGS.dataset.lower() +                                    '/example_features_nums_{}_{}.pkl'.format(FLAGS.load_small_portion, FLAGS.max_considered_history_turns)
        
    try:
        print('attempting to load train features from cache')
        with open(features_fname, 'rb') as handle:
            train_features = pickle.load(handle)
        with open(example_tracker_fname, 'rb') as handle:
            example_tracker = pickle.load(handle)
        with open(variation_tracker_fname, 'rb') as handle:
            variation_tracker = pickle.load(handle)
        with open(example_features_nums_fname, 'rb') as handle:
            example_features_nums = pickle.load(handle)
    except:
        print('train feature cache does not exist, generating')
        train_features, example_tracker, variation_tracker, example_features_nums = convert_examples_to_variations_and_then_features(
                                                            examples=train_examples, tokenizer=tokenizer, 
                                                            max_seq_length=FLAGS.max_seq_length, doc_stride=FLAGS.doc_stride, 
                                                            max_query_length=FLAGS.max_query_length, 
                                                            max_considered_history_turns=FLAGS.max_considered_history_turns, 
                                                            is_training=True)
        with open(features_fname, 'wb') as handle:
            pickle.dump(train_features, handle)
        with open(example_tracker_fname, 'wb') as handle:
            pickle.dump(example_tracker, handle)
        with open(variation_tracker_fname, 'wb') as handle:
            pickle.dump(variation_tracker, handle)     
        with open(example_features_nums_fname, 'wb') as handle:
            pickle.dump(example_features_nums, handle) 
        print('train features generated')
                
    train_batches = cqa_gen_example_aware_batches_v2(train_features, example_tracker, variation_tracker, example_features_nums, 
                                                  FLAGS.train_batch_size, FLAGS.num_train_epochs, shuffle=True)
    # temp_train_batches = cqa_gen_example_aware_batches_v2(train_features, example_tracker, variation_tracker, example_features_nums, 
    #                                               24, 1, shuffle=True)
    # print('len temp_train_batches', len(list(temp_train_batches)))
    
    # an estimation of num_train_steps
    # num_train_steps = int((math.ceil(len(train_examples) * 1.5 / FLAGS.train_batch_size)) * FLAGS.num_train_epochs)
    # we cannot predict the exact training steps because of the "example-aware" batching, 
    # so we run some initial experiments and found out the exact training steps
    # num_train_steps = 11438 * FLAGS.num_train_epochs
    num_train_steps = FLAGS.train_steps
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

if FLAGS.do_predict:
    # read in validation data, generate val features
    val_file = FLAGS.quac_predict_file
    val_examples = read_quac_examples(input_file=val_file, is_training=False)

    
    # we read in the val file in json for the external_call function in the validation step
    val_file_json = json.load(open(val_file, 'r'))['data']
    
    # we attempt to read features from cache
    features_fname = FLAGS.cache_dir + FLAGS.dataset.lower() +                                    '/val_features_{}_{}.pkl'.format(FLAGS.load_small_portion, FLAGS.max_considered_history_turns)
    example_tracker_fname = FLAGS.cache_dir + FLAGS.dataset.lower() +                                    '/val_example_tracker_{}_{}.pkl'.format(FLAGS.load_small_portion, FLAGS.max_considered_history_turns)
    variation_tracker_fname = FLAGS.cache_dir + FLAGS.dataset.lower() +                                    '/val_variation_tracker_{}_{}.pkl'.format(FLAGS.load_small_portion, FLAGS.max_considered_history_turns)
    example_features_nums_fname = FLAGS.cache_dir + FLAGS.dataset.lower() +                                    '/val_example_features_nums_{}_{}.pkl'.format(FLAGS.load_small_portion, FLAGS.max_considered_history_turns)
        
    try:
        print('attempting to load val features from cache')
        with open(features_fname, 'rb') as handle:
            val_features = pickle.load(handle)
        with open(example_tracker_fname, 'rb') as handle:
            val_example_tracker = pickle.load(handle)
        with open(variation_tracker_fname, 'rb') as handle:
            val_variation_tracker = pickle.load(handle)
        with open(example_features_nums_fname, 'rb') as handle:
            val_example_features_nums = pickle.load(handle)
    except:
        print('val feature cache does not exist, generating')
        val_features, val_example_tracker, val_variation_tracker, val_example_features_nums =                                                         convert_examples_to_variations_and_then_features(
                                                        examples=val_examples, tokenizer=tokenizer, 
                                                        max_seq_length=FLAGS.max_seq_length, doc_stride=FLAGS.doc_stride, 
                                                        max_query_length=FLAGS.max_query_length, 
                                                        max_considered_history_turns=FLAGS.max_considered_history_turns, 
                                                        is_training=False)
        with open(features_fname, 'wb') as handle:
            pickle.dump(val_features, handle)
        with open(example_tracker_fname, 'wb') as handle:
            pickle.dump(val_example_tracker, handle)
        with open(variation_tracker_fname, 'wb') as handle:
            pickle.dump(val_variation_tracker, handle)  
        with open(example_features_nums_fname, 'wb') as handle:
            pickle.dump(val_example_features_nums, handle)
        print('val features generated')
    
     
    num_val_examples = len(val_examples)
    

# tf Graph input
unique_ids = tf.placeholder(tf.int32, shape=[None], name='unique_ids')
input_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_mask')
segment_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='segment_ids')
start_positions = tf.placeholder(tf.int32, shape=[None], name='start_positions')
end_positions = tf.placeholder(tf.int32, shape=[None], name='end_positions')
history_answer_marker = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='history_answer_marker')
training = tf.placeholder(tf.bool, name='training')
get_segment_rep = tf.placeholder(tf.bool, name='get_segment_rep')
yesno_labels = tf.placeholder(tf.int32, shape=[None], name='yesno_labels')
followup_labels = tf.placeholder(tf.int32, shape=[None], name='followup_labels')

# a unique combo of (e_tracker, f_tracker) is called a slice
slice_mask = tf.placeholder(tf.int32, shape=[FLAGS.train_batch_size, ], name='slice_mask') 
slice_num = tf.placeholder(tf.int32, shape=None, name='slice_num') 

# for auxiliary loss
aux_start_positions = tf.placeholder(tf.int32, shape=[None], name='aux_start_positions')
aux_end_positions = tf.placeholder(tf.int32, shape=[None], name='aux_end_positions')

bert_representation, cls_representation = bert_rep(
        bert_config=bert_config,
        is_training=training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        history_answer_marker=history_answer_marker,
        use_one_hot_embeddings=False
        )

reduce_mean_representation = tf.reduce_mean(bert_representation, axis=1)
reduce_max_representation = tf.reduce_max(bert_representation, axis=1) 

if FLAGS.history_attention_input == 'CLS':
    history_attention_input = cls_representation    
elif FLAGS.history_attention_input == 'reduce_mean':
    history_attention_input = reduce_mean_representation
elif FLAGS.history_attention_input == 'reduce_max':
    history_attention_input = reduce_max_representation
else:
    print('FLAGS.history_attention_input not specified')
    
if FLAGS.mtl_input == 'CLS':
    mtl_input = cls_representation    
elif FLAGS.mtl_input == 'reduce_mean':
    mtl_input = reduce_mean_representation
elif FLAGS.mtl_input == 'reduce_max':
    mtl_input = reduce_max_representation
else:
    print('FLAGS.mtl_input not specified')
    

if FLAGS.aux_shared:
    # if the aux prediction layer is shared with the main convqa model:
    (aux_start_logits, aux_end_logits) = cqa_model(bert_representation)
else:
    # if they are not shared
    (aux_start_logits, aux_end_logits) = aux_cqa_model(bert_representation)




if FLAGS.disable_attention:
    new_bert_representation, new_mtl_input, attention_weights = disable_history_attention_net(bert_representation, 
                                                                                     history_attention_input, mtl_input, 
                                                                                     slice_mask,
                                                                                     slice_num)

else:
    if FLAGS.fine_grained_attention:
        new_bert_representation, new_mtl_input, attention_weights = fine_grained_history_attention_net(bert_representation, 
                                                                                         mtl_input,
                                                                                         slice_mask,
                                                                                         slice_num)

    else:
        new_bert_representation, new_mtl_input, attention_weights = history_attention_net(bert_representation, 
                                                                                         history_attention_input, mtl_input,
                                                                                         slice_mask,
                                                                                         slice_num)

(start_logits, end_logits) = cqa_model(new_bert_representation)
yesno_logits = yesno_model(new_mtl_input)
followup_logits = followup_model(new_mtl_input)

tvars = tf.trainable_variables()
# print(tvars)

initialized_variable_names = {}
if FLAGS.init_checkpoint:
    (assignment_map, initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
    tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
# print('tvars',tvars)
# print('initialized_variable_names',initialized_variable_names)

# compute loss
seq_length = modeling.get_shape_list(input_ids)[1]
def compute_loss(logits, positions):
    one_hot_positions = tf.one_hot(
        positions, depth=seq_length, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
    return loss

# get the max prob for the predicted start/end position
start_probs = tf.nn.softmax(start_logits, axis=-1)
start_prob = tf.reduce_max(start_probs, axis=-1)
end_probs = tf.nn.softmax(end_logits, axis=-1)
end_prob = tf.reduce_max(end_probs, axis=-1)

start_loss = compute_loss(start_logits, start_positions)
end_loss = compute_loss(end_logits, end_positions)

yesno_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yesno_logits, labels=yesno_labels))
followup_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=followup_logits, labels=followup_labels))

if FLAGS.MTL:
    cqa_loss = (start_loss + end_loss) / 2.0
    if FLAGS.MTL_lambda < 1:
        total_loss = FLAGS.MTL_mu * cqa_loss +                      FLAGS.MTL_lambda * yesno_loss +                      FLAGS.MTL_lambda * followup_loss
    else:
        total_loss = cqa_loss + yesno_loss + followup_loss
    tf.summary.scalar('cqa_loss', cqa_loss)
    tf.summary.scalar('yesno_loss', yesno_loss)
    tf.summary.scalar('followup_loss', followup_loss)
else:
    total_loss = (start_loss + end_loss) / 2.0


# if FLAGS.aux:
#     aux_start_probs = tf.nn.softmax(aux_start_logits, axis=-1)
#     aux_start_prob = tf.reduce_max(aux_start_probs, axis=-1)
#     aux_end_probs = tf.nn.softmax(aux_end_logits, axis=-1)
#     aux_end_prob = tf.reduce_max(aux_end_probs, axis=-1)
#     aux_start_loss = compute_loss(aux_start_logits, aux_start_positions)
#     aux_end_loss = compute_loss(aux_end_logits, aux_end_positions)
    
#     aux_loss = (aux_start_loss + aux_end_loss) / 2.0
#     cqa_loss = (start_loss + end_loss) / 2.0
#     total_loss = (1 - FLAGS.aux_lambda) * cqa_loss + FLAGS.aux_lambda * aux_loss
    
#     tf.summary.scalar('cqa_loss', cqa_loss)
#     tf.summary.scalar('aux_loss', aux_loss)

# else:
#     total_loss = (start_loss + end_loss) / 2.0


tf.summary.scalar('total_loss', total_loss)

if FLAGS.do_train:
    train_op = optimization.create_optimizer(total_loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, False)

    print("***** Running training *****")
    print("  Num orig examples = %d", len(train_examples))
    print("  Num train_features = %d", len(train_features))
    print("  Batch size = %d", FLAGS.train_batch_size)
    print("  Num steps = %d", num_train_steps)
    
merged_summary_op = tf.summary.merge_all()

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits", "yesno_logits", "followup_logits"])

attention_dict = {}

saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()
tf.get_default_graph().finalize()
with tf.Session() as sess:
    sess.run(init)

    if FLAGS.do_train:
        train_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/train', sess.graph)
        val_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/val')
        rl_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/rl')
        
        f1_list = []
        heq_list = []
        dheq_list = []
        yesno_list, followup_list = [], []
        
        # Training cycle
        for step, batch in enumerate(train_batches):
            if step > num_train_steps:
                # this means the learning rate has been decayed to 0
                break
            
            # time1 = time()
            batch_features, batch_slice_mask, batch_slice_num, output_features = batch                      

            fd = convert_features_to_feed_dict(batch_features) # feed_dict
            fd_output = convert_features_to_feed_dict(output_features)
            
            if FLAGS.better_hae:
                turn_features = get_turn_features(fd['metadata'])
                fd['history_answer_marker'] = fix_history_answer_marker_for_bhae(fd['history_answer_marker'], turn_features)
            
            if FLAGS.history_ngram != 1:
                batch_slice_mask, group_batch_features = group_histories(batch_features, fd['history_answer_marker'], 
                                                                                  batch_slice_mask, batch_slice_num)
                fd = convert_features_to_feed_dict(group_batch_features)
            
            
            try:
                _, train_summary, total_loss_res = sess.run([train_op, merged_summary_op, total_loss], 
                                                feed_dict={unique_ids: fd['unique_ids'], input_ids: fd['input_ids'], 
                                                input_mask: fd['input_mask'], segment_ids: fd['segment_ids'], 
                                                start_positions: fd_output['start_positions'], end_positions: fd_output['end_positions'], 
                                                history_answer_marker: fd['history_answer_marker'], slice_mask: batch_slice_mask, 
                                                slice_num: batch_slice_num, 
                                                aux_start_positions: fd['start_positions'], aux_end_positions: fd['end_positions'],
                                                yesno_labels: fd_output['yesno'], followup_labels: fd_output['followup'], training: True})
            except Exception as e:
                print('training, features length: ', len(batch_features))
                print(e)
                traceback.print_tb(e.__traceback__)

            train_summary_writer.add_summary(train_summary, step)
            train_summary_writer.flush()
            # print('attention weights', attention_weights_res)
            print('training step: {}, total_loss: {}'.format(step, total_loss_res))
            # time2 = time()
            # print('train step', time2-time1)
            
            
            # if (step % 3000 == 0 or                 (step >= FLAGS.evaluate_after and step % FLAGS.evaluation_steps == 0)) and                 step != 0:
            if step >= FLAGS.evaluate_after and step % FLAGS.evaluation_steps == 0:
                    
                val_total_loss = []
                all_results = []
                all_output_features = []
                
                val_batches = cqa_gen_example_aware_batches_v2(val_features, val_example_tracker, val_variation_tracker, 
                                                  val_example_features_nums, FLAGS.predict_batch_size, 1, shuffle=False)
                
                for val_batch in val_batches:
                    # time3 = time()
                    batch_results = []
                    batch_features, batch_slice_mask, batch_slice_num, output_features = val_batch
                        
                    try:
                        all_output_features.extend(output_features)

                        fd = convert_features_to_feed_dict(batch_features) # feed_dict
                        fd_output = convert_features_to_feed_dict(output_features)

                        if FLAGS.better_hae:
                            turn_features = get_turn_features(fd['metadata'])
                            fd['history_answer_marker'] = fix_history_answer_marker_for_bhae(fd['history_answer_marker'], turn_features)
                            
                        if FLAGS.history_ngram != 1:                     
                            batch_slice_mask, group_batch_features = group_histories(batch_features, fd['history_answer_marker'], 
                                                                                  batch_slice_mask, batch_slice_num)
                            fd = convert_features_to_feed_dict(group_batch_features)
                        
                        start_logits_res, end_logits_res,                         yesno_logits_res, followup_logits_res,                         batch_total_loss,                         attention_weights_res = sess.run([start_logits, end_logits, yesno_logits, followup_logits, 
                                                          total_loss, attention_weights], 
                                    feed_dict={unique_ids: fd['unique_ids'], input_ids: fd['input_ids'], 
                                    input_mask: fd['input_mask'], segment_ids: fd['segment_ids'], 
                                    start_positions: fd_output['start_positions'], end_positions: fd_output['end_positions'], 
                                    history_answer_marker: fd['history_answer_marker'], slice_mask: batch_slice_mask, 
                                    slice_num: batch_slice_num,
                                    aux_start_positions: fd['start_positions'], aux_end_positions: fd['end_positions'],
                                    yesno_labels: fd_output['yesno'], followup_labels: fd_output['followup'], training: False})

                        val_total_loss.append(batch_total_loss)
                        
                        key = (tuple([val_examples[f.example_index].qas_id for f in output_features]), step)
                        attention_dict[key] = {'batch_slice_mask': batch_slice_mask, 'attention_weights_res': attention_weights_res,
                                              'batch_slice_num': batch_slice_num, 'len_batch_features': len(batch_features),
                                              'len_output_features': len(output_features)}

                        for each_unique_id, each_start_logits, each_end_logits, each_yesno_logits, each_followup_logits                                 in zip(fd_output['unique_ids'], start_logits_res, end_logits_res, yesno_logits_res, followup_logits_res):  
                            each_unique_id = int(each_unique_id)
                            each_start_logits = [float(x) for x in each_start_logits.flat]
                            each_end_logits = [float(x) for x in each_end_logits.flat]
                            each_yesno_logits = [float(x) for x in each_yesno_logits.flat]
                            each_followup_logits = [float(x) for x in each_followup_logits.flat]
                            batch_results.append(RawResult(unique_id=each_unique_id, start_logits=each_start_logits, 
                                                           end_logits=each_end_logits, yesno_logits=each_yesno_logits,
                                                          followup_logits=each_followup_logits))

                        all_results.extend(batch_results)
                    except Exception as e:
                        print('batch dropped because too large!')
                        print('validating, features length: ', len(batch_features))
                        print(e)
                        traceback.print_tb(e.__traceback__)
                    # time4 = time()
                    # print('val step', time4-time3)
                output_prediction_file = os.path.join(FLAGS.output_dir, "predictions_{}.json".format(step))
                output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions_{}.json".format(step))
                output_null_log_odds_file = os.path.join(FLAGS.output_dir, "output_null_log_odds_file_{}.json".format(step))
                
                # time5 = time()
                write_predictions(val_examples, all_output_features, all_results,
                                  FLAGS.n_best_size, FLAGS.max_answer_length,
                                  FLAGS.do_lower_case, output_prediction_file,
                                  output_nbest_file, output_null_log_odds_file)
                # time6 = time()
                # print('write all val predictions', time6-time5)
                val_total_loss_value = np.average(val_total_loss)
                                
                
                # call the official evaluation script
                val_summary = tf.Summary() 
                # time7 = time()
                val_eval_res = external_call(val_file_json, output_prediction_file)
                # time8 = time()
                # print('external call', time8-time7)
                val_f1 = val_eval_res['f1']
                val_followup = val_eval_res['followup']
                val_yesno = val_eval_res['yes/no']
                val_heq = val_eval_res['HEQ']
                val_dheq = val_eval_res['DHEQ']

                heq_list.append(val_heq)
                dheq_list.append(val_dheq)
                yesno_list.append(val_yesno)
                followup_list.append(val_followup)

                val_summary.value.add(tag="followup", simple_value=val_followup)
                val_summary.value.add(tag="val_yesno", simple_value=val_yesno)
                val_summary.value.add(tag="val_heq", simple_value=val_heq)
                val_summary.value.add(tag="val_dheq", simple_value=val_dheq)

                print('evaluation: {}, total_loss: {}, f1: {}, followup: {}, yesno: {}, heq: {}, dheq: {}\n'.format(
                    step, val_total_loss_value, val_f1, val_followup, val_yesno, val_heq, val_dheq))
                with open(FLAGS.output_dir + 'step_result.txt', 'a') as fout:
                        fout.write('{},{},{},{},{},{},{}\n'.format(step, val_f1, val_heq, val_dheq, 
                                                                   val_yesno, val_followup, FLAGS.output_dir))
                
                val_summary.value.add(tag="total_loss", simple_value=val_total_loss_value)
                val_summary.value.add(tag="f1", simple_value=val_f1)
                f1_list.append(val_f1)
                val_summary_writer.add_summary(val_summary, step)
                val_summary_writer.flush()
                
                save_path = saver.save(sess, '{}/model_{}.ckpt'.format(FLAGS.output_dir, step))
                print('Model saved in path', save_path)

                



# In[4]:


best_f1 = max(f1_list)
best_f1_idx = f1_list.index(best_f1)
best_heq = heq_list[best_f1_idx]
best_dheq = dheq_list[best_f1_idx]
best_yesno = yesno_list[best_f1_idx]
best_followup = followup_list[best_f1_idx]
with open(FLAGS.output_dir + 'result.txt', 'w') as fout:
    fout.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(best_f1, best_heq, best_dheq, best_yesno, best_followup,
                                                  FLAGS.MTL_lambda, FLAGS.MTL_mu, FLAGS.MTL, FLAGS.mtl_input,
                                                  FLAGS.history_attention_input, FLAGS.output_dir))

