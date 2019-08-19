# Attentive History Selection for Conversational Question Answering

This is the implementation for the "HAM" model proposed in the CIKM'19 paper [Attentive History Selection for Conversational Question Answering](xxx). This model first incorporates history turns with positional history answer embedding (PosHAE) with a [BERT](https://github.com/google-research/bert) based model, and then conducts soft selection of history by attending to the history turns.

If you use this code for your paper, please cite it as  
```
Chen Qu, Liu Yang, Minghui Qiu, Yongfeng Zhang, Cen Chen, W. Bruce Croft and Mohit Iyyer. Attentive History Selection for Conversational Question Answering. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM 2019), Beijing, China, November 03-07, 2019.

Bibtext
@inproceedings{ham,
	author = {Qu, C. and Yang, L. and Qiu, M. and Zhang, Y. and Chen, C. and Croft, W. B. and Iyyer, M.},
	title = {Attentive History Selection for Conversational Question Answering},
	booktitle = {CIKM '19},
	year = {2019},
}
```

### Run

1. Download the `BERT-base Uncased` model [here](https://github.com/google-research/bert).
2. Download the [QuAC](http://quac.ai/) data.
3. Configurate the directories for the BERT model, data, output, and cache in `cqa_flags.py`. 
4. Run 

```
python cqa_run_his_atten.py 
    --output_dir=OUTPUT_DIR\
    --max_considered_history_turns=11 \
    --num_train_epochs=15.0 \
    --train_steps=30000 \
    --learning_rate=3e-5 \
    --n_best_size=20 \
    --better_hae=True \
    --MTL=False \
    --MTL_lambda=0.0 \
    --train_batch_size=24 \
    --predict_batch_size=24 \
    --evaluate_after=24000 \
    --evaluation_steps=1000 \
    --disable_attention=False \
    --aux=False \
    --aux_shared=False \
    --aux_lambda=0.0 \
    --history_attention_hidden=False \
    --fine_grained_attention=True \
    --history_ngram=1
```
Setting the max_seq_length to 512 should give better results.

### Some program arguments

Program arguments can be set in `cqa_flgas.py`. Alternatively, they could be specified at running by command line arguments like above. Most of the arguments are self-explanatory. Here are some selected arguments:

* `num_train_epochs` , `train_steps`,`learning_rate` ,`warmup_proportion`: the learning rate follow a schedule of warming up to the specified larning rate and then decaying. This schedule is described in the transformer paper. Our model trains for `train_steps` instead of full `num_train_epochs` epochs. 
* `load_small_portion` . Set to `True` for loading a small portion of the data for testing purpose when we are developing the model. Set to `False` to load all the data when running the model.
* `cache_dir`. When we run the model for the first time, it preprocesses the data and saves it in a cache directory. After that, the model reads the propocessed data from the cache.
* `max_considered_history_turns` and `max_history_turns`. We only consider `max_considered_history_turns` previous turns when preprocessing the data. This is typically set to 11, meaning that all previous turns are under consideration (for QuAC). The `max_history_turns` is for padding purpose in the history attention module.

For other arguments, please kindly refer to `cqa_flgas.py` for help messages.


### Scripts

* `cqa_run_his_atten.py`. Entry code.
* `cqa_supports.py`. Utility functions.
* `cqa_gen_batches.py`. Generate batches.
* `cqa_model.py`. Our models.
* `scorer.py`. Official evaluation script for QuAC.

Most other files are for BERT.


### Environment

Tested with Python 3.6.7 and TensorFlow 1.8.0
