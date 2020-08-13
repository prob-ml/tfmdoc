import os
import sys
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import torch
import pandas as pd
import random
from torch.distributions.binomial import Binomial
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import DataCollatorForLanguageModeling, BertForMaskedLM
from transformers import Trainer, TrainingArguments

from tokens import WordLevelBertTokenizer
from vocab import create_vocab
from data import CausalBertDataset, MLMDataset


class CausalBert(nn.Module):
    def __init__(self, token_embed, hidden_size=256, binary_response=True, learnable_docu_embed=False):
        super().__init__()
        self.token_embed = token_embed
        self.learnable_docu_embed = learnable_docu_embed

        embed_out = self.token_embed.weight.shape[1]
        
        # A learnable sum of embed to document.
        if self.learnable_docu_embed:
            max_length = self.token_embed.weight.shape[0]
            self.docu_embed = nn.Linear(max_length, 1)
            
        # G head: logit-linear mapping
        self.g = nn.Sequential(*[nn.Linear(embed_out, 1), nn.Sigmoid()])

        # Q_1 and Q_0 heads: two-hidden layers mapping.
        if binary_response:
            self.q1 = nn.Sequential(
                *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), nn.Sigmoid()])
            self.q0 = nn.Sequential(
                *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), nn.Sigmoid()])
        else:
            self.q1 = nn.Sequential(
                *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), ])
            self.q0 = nn.Sequential(
                *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), ])

    def forward(self, tokens):
        embed_token = self.token_embed(tokens)
        
        if self.learnable_docu_embed:
            bsz = embed_token.shape[0]
            max_len = embed_token.shape[1]
            embed_token = embed_token.permute(0, 2, 1).reshape(-1, max_len)
            embed_docu = self.docu_embed(embed_token)
            embed_docu = embed_docu.reshape(bsz, -1)
        else:
            embed_docu = torch.sum(embed_token, axis=1)

        return self.g(embed_docu), self.q1(embed_docu), self.q0(embed_docu)


def causal_bert_task(args):
    pass


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
