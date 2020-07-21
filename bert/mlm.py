import os
import sys
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from transformers import DataCollatorForLanguageModeling, BertForMaskedLM
from transformers import Trainer, TrainingArguments

from data import LineByLineTextDataset
from tokens import WordLevelBertTokenizer
from vocab import create_vocab
from utils import DATA_PATH, make_dirs


def mlm_task(args):
    merged = (args.data == 'merged')

    print('Start: collect vocab of EMR.')
    vocab = create_vocab(merged=merged)
    print('Finish: collect vocab of EMR.')
    print('*' * 200)

    print('Start: load word level tokenizer.')
    tokenizer = WordLevelBertTokenizer(vocab)
    print('Finish: load word level tokenizer.')
    print('*' * 200)

    print('Start: load data (and encode to token sequence.)')
    dataset = LineByLineTextDataset(tokenizer=tokenizer, data_type=args.data,
                                    max_length=args.max_length, min_length=args.min_length,
                                    truncate_method=args.truncate)
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
    print(f'Bert model: contains {model.num_parameters()} parameters.')

    if args.model == 'dev':
        training_args = TrainingArguments(output_dir=result_path, overwrite_output_dir=True,
                                          num_train_epochs=1,
                                          per_device_train_batch_size=args.bsz,
                                          save_steps=10_000, )
    if args.model == 'behrt':
        training_args = TrainingArguments(output_dir=result_path, overwrite_output_dir=True,
                                          num_train_epochs=100,
                                          per_device_train_batch_size=32,
                                          save_steps=10_000, )

    if args.model == 'med-bert':
        total_step = 450_0000
        epoch_step = len(dataset) // args.bsz
        epoch = total_step / epoch_step
        training_args = TrainingArguments(output_dir=result_path, overwrite_output_dir=True,
                                          num_train_epochs=epoch,
                                          per_device_train_batch_size=args.bsz,
                                          save_steps=10_000, )

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=mlm_collator,
                      train_dataset=dataset,
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
    parser.add_argument('--truncate', type=str, choices=['first', 'last', 'random'], default='first')
    parser.add_argument('--min-length', type=int, default=10, help='Min length of a sequence to be used in Bert')
    parser.add_argument('--max-length', type=int, default=512, help='Max length of a sequence used in Bert')
    parser.add_argument('--bsz', type=int, default=3, help='Batch size in training')
    parser.add_argument('--epochs', type=int, default=10, help='Epoch in production version')
    parser.add_argument('--force-new', action='store_true', default=False, help='Force to train a new MLM.')
    parser.add_argument('--model', type=str, default='behrt', choices=['dev', 'behrt', 'med-bert'],
                        help='Run dev version to make sure codes can run.')
    parser.add_argument('--cuda', type=str, help='Visible CUDA to the task.')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(f'Prepare: check process on cuda: {args.cuda}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_path = os.path.join(curPath, 'results', args.model, 'MLM')
    trained_model = os.path.join(curPath, 'trained', args.model, 'MLM')
    make_dirs(result_path, trained_model)

    assert args.model in ['dev', 'behrt', 'med-bert'], f'Not supported for model config: {args.model} yet...'
    if args.model == 'med-bert':
        raise UserWarning('Configuration of Med-Bert from the paper is still mysterious, '
                          'the final result may be unexpected...')
    mlm_task(args)
    print('Finish all...')


