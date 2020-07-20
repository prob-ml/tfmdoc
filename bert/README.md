## Highlighted Milestone:

- [ ] Use multiple GPUs to train a large model like Bert-base.
- [ ] Add better ways to encode daily token sequence, e.g.,  Deep set.
- [ ] Extract contextual embedding of tokens part from Bert which can be fine-tuned with different downstream tasks, e.g., Causal-Bert.
- [x] Practical guide for training Bert-base on a Single GPU GTX 2080 Ti with 11 GB memory.
- [x] A runable pipeline for Pre-train Bert-base on EMR data.

## Term explanation

Here are explicit explanations of some basic terms we will use:

- *Pre-train*: use MLM to learn contextual embedding.
- *Fine-tune*: 
  - In Causal-Bert, it refers to fine tune the layers in Bert producing the contextual embeddings, and three heads in Dragonnet.
  - In other usages: it may refer to fine tune the whole Bert with other applications. 
- *Bert-base*: the **cased** 12-layer, 768-hidden, 12-heads, 110M parameters, note: we didn't explicitly transforming all characters to lower case currently(2020/07/16). 

## Overview of the project

There are several main scripts:

`mlm.py` : pre-train Bert with masked LM task, and we inlclude the default parameter values below

```bash
python3 mlm.py \
	--data  merged # (or, daily)
	--truncate first # or last, random, specify different ways to truncate a sequence longer than max-length.
	--max-length 512 # See 'Batch size and max sequence length' section for other choices.
	--bsz 3 # See 'Batch size and max sequence length' section for other choices.
	--epochs 10 # in dev mode, it will be 1
	--force-new False # Always pretrain a new Bert.
	--dev False # A dev mode includes smaller Bert and smaller epochs, check if the codes can run.
	--cuda '4' # or, '4,5,6,7' to allow Bert see GPU 4~7.
```

TODO

- [ ] Fine-tune Causal Bert

- [ ] Fine-tune with other application

  

Useful tips: 

- In the cmd, you only need to include parameters which you want to use other values than default. For example, if you want to train a Bert-base on `CUDA 5` with `max-length 256`, just run `python3 mlm.py --cuda 5 --max-length 256`. 
- You may want to use `nohup` to let task keep running on the server after logout, In order to do so, just run `nohup python3 mlm.py > a_path_to_save_log 2>&1 & `, and here are brief explanations of this command:
  - ` > a_path_to_save_log` redirects the output (logs) to the `a_path_save_log`
  - `2>&1` adds `stderr`(2) to `stdout`(1), so that the error messages will also be printed (to the output(logs))
  - `&` let the task run in the background so that you can keep use your command line.

## Batch size and max sequence length

Based on the offical guide regarding [out-of-memory-issues](https://github.com/google-research/bert#out-of-memory-issues) of Bert, our single GPU can afford 

| System    | Max Seq Length | Max Batch Size |
| --------- | -------------- | -------------- |
| Bert-base | 256            |                |
|	    | 384	     | 5              |
|           | 512            | 2              |



## Role of other scripts explanations

Finally, we include a full list of all scripts and their roles:

- `__init__.py`: *I (@Tianci Liu) don't use this file as well, and I am not sure when to use this scirpts yet.*
- `vocab.py`: this script contains a function retrieving the path of `vocab.json`, which depends on the input data (e.g., the user group, or data type: daily/merged.).
- `tokens.py`: this script contains a `WordLevelBertTokenizer` (and a `WordLevelTokenizer` which should be always used with `WordLevelBertTokenizer`), its input is the path of the `vocab.json`. 
- `data.py`: this script contains a line-by-line text dataset class, whose input requires WordLevelBertTokenizer.
- `utils.py`: this script contains some tool functions and constant like `DATA_PATH`. 
