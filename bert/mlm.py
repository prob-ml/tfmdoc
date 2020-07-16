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

    vocab = create_vocab(merged=merged)
    tokenizer = WordLevelBertTokenizer(vocab)

    dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=DATA_PATH, max_length=args.max_length)
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, )

    if not os.listdir(trained_model) or args.force_new:
        print('Train a new model...')

        from transformers import BertConfig
        if args.dev:
            config = BertConfig(vocab_size=len(tokenizer), max_position_embeddings=args.max_length,
                                num_attention_heads=2,
                                num_hidden_layers=4,
                                hidden_size=128,
                                type_vocab_size=1, )
        else:
            config = BertConfig(vocab_size=len(tokenizer), max_position_embeddings=args.max_length,
                                num_attention_heads=12,
                                num_hidden_layers=12,
                                hidden_size=768,
                                type_vocab_size=1, )

    else:
        print('Load a trained model...')

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(trained_model)

    model = BertForMaskedLM(config=config)
    print(f'Bert model: contains {model.num_parameters()} parameters.')

    if args.dev:
        training_args = TrainingArguments(output_dir=result_path, overwrite_output_dir=True,
                                          num_train_epochs=1,
                                          per_device_train_batch_size=args.bsz,
                                          save_steps=10_000,)
    else:
        training_args = TrainingArguments(output_dir=result_path, overwrite_output_dir=True,
                                          num_train_epochs=args.epochs,
                                          per_device_train_batch_size=args.bsz,
                                          save_steps=10_000, )

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=mlm_collator,
                      train_dataset=dataset,
                      prediction_loss_only=True, )

    print('Start: pre-train Bert with MLM.')
    trainer.train()
    print('Finish: pre-train Bert with MLM.')

    print('Start: save pre-train Bert with MLM.')
    trainer.save(trained_model)
    print('Finish: save pre-train Bert with MLM.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['daily', 'merged'], default='merged')
    parser.add_argument('--max-length', type=int, default=512, help='Max length of a sequence used in Bert')
    parser.add_argument('--bsz', type=int, default=64, help='Batch size in training')
    parser.add_argument('--epochs', type=int, default=10, help='Epoch in production version')

    parser.add_argument('--force-new', action='store_true', default=False, help='Force to train a new MLM.')
    parser.add_argument('--dev', action='store_true', default=True, help='Run dev version to make sure codes can run.')
    parser.add_argument('--cuda', type=str, default='5,6,7', help='Visible CUDA to the task.')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda if not args.dev else args.cuda[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_path = os.path.join(curPath, 'results' if not args.dev else 'result-dev', 'MLM')
    trained_model = os.path.join(curPath, 'trained' if not args.dev else 'trained-dev', 'MLM')
    make_dirs(result_path, trained_model)

    mlm_task(args)

