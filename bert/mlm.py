import os
import sys
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from transformers import DataCollatorForLanguageModeling, BertForMaskedLM
from transformers import Trainer, TrainingArguments

from data import MLMDataset
from tokens import WordLevelBertTokenizer
from vocab import create_vocab
from utils import DATA_PATH, make_dirs


def mlm_task(args):
    merged = (args.data == 'merged')

    print('Start: collect vocab of EMR.')
    vocab = create_vocab(merged=merged, uni_diag=args.unidiag)
    print('Finish: collect vocab of EMR.')
    print('*' * 200)

    print('Start: load word level tokenizer.')
    tokenizer = WordLevelBertTokenizer(vocab)
    print(f'Finish: load word level tokenizer. Vocab size: {len(tokenizer)}.')
    print('*' * 200)

    print('Start: load data (and encode to token sequence.)')

    # If want to evaluate model during trainning.
    if args.eval_when_train:
        train_group = list(range(9))
        eval_group = [9]
        train_dataset = MLMDataset(tokenizer=tokenizer, data_type=args.data, is_unidiag=args.unidiag,
                                              group=train_group, max_length=args.max_length, min_length=args.min_length,
                                              truncate_method=args.truncate, device=device)
        eval_dataset = MLMDataset(tokenizer=tokenizer, data_type=args.data, is_unidiag=args.unidiag,
                                              group=eval_group, max_length=args.max_length, min_length=args.min_length,
                                              truncate_method=args.truncate, device=device)
    else:
        train_dataset = MLMDataset(tokenizer=tokenizer, data_type=args.data, is_unidiag=args.unidiag,
                                              max_length=args.max_length, min_length=args.min_length,
                                              truncate_method=args.truncate, device=device)
        eval_dataset = None

    print('Finish: load data (and encode to token sequence.)')
    print('*' * 200)

    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, )

    if not os.listdir(trained_model) or args.force_new:
        print('Train a new model...')

        from transformers import BertConfig
        if args.model == 'dev':
            config = BertConfig(vocab_size=len(tokenizer), max_position_embeddings=args.max_length,
                                num_attention_heads=2,
                                num_hidden_layers=4,
                                hidden_size=128,
                                type_vocab_size=1, )
            
        if args.model == 'behrt':
            config = BertConfig(vocab_size=len(tokenizer), max_position_embeddings=args.max_length,
                                num_attention_heads=12,
                                num_hidden_layers=6,
                                intermediate_size=512,
                                hidden_size=288,
                                type_vocab_size=1, )

        if args.model == 'med-bert':
            config = BertConfig(vocab_size=len(tokenizer), max_position_embeddings=args.max_length,
                                num_attention_heads=6,
                                num_hidden_layers=6,
                                intermediate_size=32,
                                hidden_size=32,
                                type_vocab_size=1, )

    else:
        print('Load a trained model...')

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(trained_model)

    model = BertForMaskedLM(config=config)
    print(f'Bert model: contains {model.num_parameters()} parameters...')
    print(f'Data set: contains {len(train_dataset)} samples.')

    epoch_step = len(train_dataset) // args.bsz
    if args.model == 'dev':
        epoch = 1
    elif args.model == 'behrt':
        # Note: in Bert base, pretrain is: bsz = 256, total iterations is 1_000_000
        epoch = 30
    elif args.model == 'med-bert':
        total_step = 4_500_000
        epoch = total_step // args.bsz

    training_args = TrainingArguments(output_dir=result_path, overwrite_output_dir=True,
                                      num_train_epochs=epoch,
                                      per_device_train_batch_size=args.bsz,
                                      evaluate_during_training=True if args.eval_when_train else False,
                                      save_total_limit=10,
                                      save_steps=epoch_step,
                                      )

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=mlm_collator,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      prediction_loss_only=True,
                      )

    print('Start: pre-train Bert with MLM.')
    trainer.train()
    print('Finish: pre-train Bert with MLM.')

    print(f'Start: save pre-train Bert with MLM to: {trained_model}.')
    trainer.save(trained_model)
    print(f'Finish: save pre-train Bert with MLM to: {trained_model}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['daily', 'merged'], default='merged')
    parser.add_argument('--diag', type=str, choices=['uni', 'raw'], default='uni',
                        help='Which tokens to use: uni for Uni-diag code, and raw for raw data.')

    # parser.add_argument('--create-vocab', action='stro, choices=['daily', 'merged'], default='merged')
    parser.add_argument('--truncate', type=str, choices=['first', 'last', 'random'], default='first')
    parser.add_argument('--min-length', type=int, default=10, help='Min length of a sequence to be used in Bert')
    parser.add_argument('--max-length', type=int, default=512, help='Max length of a sequence used in Bert')
    parser.add_argument('--bsz', type=int, default=8, help='Batch size in training')
    parser.add_argument('--epochs', type=int, default=10, help='Epoch in production version')
    parser.add_argument('--force-new', action='store_true', default=False, help='Force to train a new MLM.')
    parser.add_argument('--model', type=str, choices=['dev', 'behrt', 'med-bert'], default='behrt',
                        help='Which bert to use')
    parser.add_argument('--cuda', type=str, help='Visible CUDA to the task.')

    parser.add_argument('--no-eval', action='store_true', default=False,
                        help='Do not evaluate during training.')

    args = parser.parse_args()

    args.unidiag = (args.diag == 'uni')
    args.eval_when_train = not args.no_eval

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(f'Prepare: check process on cuda: {args.cuda}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_path = os.path.join(DATA_PATH, 'results', args.model, 'MLM', args.data, 'unidiag' if args.unidiag else 'original' )
    trained_model = os.path.join(curPath, 'trained', args.model, 'MLM', 'unidiag' if args.unidiag else 'original')

    result_path = os.path.join(curPath, 'results', args.model, 'MLM', args.data, 'unidiag' if args.unidiag else 'original' )
    trained_model = os.path.join(curPath, 'trained', args.model, 'MLM', 'unidiag' if args.unidiag else 'original')
    make_dirs(result_path, trained_model)
    print(f'Prepare: check result at {result_path}.')

    assert args.model in ['dev', 'behrt', 'med-bert'], f'Not supported for model config: {args.model} yet...'
    if args.model == 'med-bert':
        raise UserWarning('Configuration of Med-Bert from the paper is still mysterious, '
                          'the final result may be unexpected...')
    mlm_task(args)

    print('Finish all...')
