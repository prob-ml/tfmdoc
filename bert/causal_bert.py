import os
import sys
import argparse
from tqdm.auto import tqdm, trange

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

from transformers import DataCollatorForLanguageModeling, BertForMaskedLM, BertModel
from transformers import Trainer, TrainingArguments

from tokens import WordLevelBertTokenizer
from vocab import create_vocab
from data import CausalBertDataset, MLMDataset
from utils import DATA_PATH, make_dirs


class CausalBOW(nn.Module):
    """
    Use a WOB embedding architecture: this is a word-of-bag model where we take the sum (or equivalently, average) of
    token input embedding without positional information.
    """
    def __init__(self, token_embed, hidden_size=256, max_length=512, binary_response=True, learnable_docu_embed=False, prop_is_logit=False):
        super().__init__()
        self.token_embed = token_embed
        self.learnable_docu_embed = learnable_docu_embed

        embed_out = self.token_embed.weight.shape[1]
        
        # A learnable sum of embed to document.
        if self.learnable_docu_embed:
            self.docu_embed = nn.Linear(max_length, 1)
            
        # G head: logit-linear mapping
        seq = [nn.Linear(embed_out, 1)]
        if not prop_is_logit:
            seq += [nn.Sigmoid()]
        self.g = nn.Sequential(*seq)

        # Q_1 and Q_0 heads: two-hidden layers mapping.
        seq = [nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True), nn.Linear(hidden_size, 1)]
        if binary_response:
            seq += [nn.Sigmoid()]
        self.q1 = nn.Sequential(*seq)
        
        seq = [nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True), nn.Linear(hidden_size, 1)]
        if binary_response:
            seq += [nn.Sigmoid()]
        self.q0 = nn.Sequential(*seq)
        
        self.prop_is_logit = prop_is_logit
#         if binary_response:
#             self.q1 = nn.Sequential(
#                 *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
#                     nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), nn.Sigmoid()])
#             self.q0 = nn.Sequential(
#                 *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
#                     nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), nn.Sigmoid()])
#         else:
#             self.q1 = nn.Sequential(
#                 *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
#                     nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), ])
#             self.q0 = nn.Sequential(
#                 *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
#                     nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), ])

    def forward(self, tokens):
        embed_token = self.token_embed(tokens)
        
        if self.learnable_docu_embed:
            bsz = embed_token.shape[0]
            max_len = embed_token.shape[1]
            embed_token = embed_token.permute(0, 2, 1).reshape(-1, max_len)
            embed_docu = self.docu_embed(embed_token)
            embed_docu = embed_docu.view(bsz, -1)
        else:
            embed_docu = torch.mean(embed_token, axis=1)

        return self.g(embed_docu), self.q1(embed_docu), self.q0(embed_docu)


class CausalBert(nn.Module):
    """
    Use a Bert last hidden-state embedding architecture,
    see discussion here for more details: https://github.com/huggingface/transformers/issues/1950
    """
    def __init__(self, bert, hidden_size=256, max_length=512, binary_response=True, learnable_docu_embed=False):
        super().__init__()
        self.bert = bert
        self.learnable_docu_embed = learnable_docu_embed

        embed_out = self.bert.get_input_embeddings().weight.shape[1]

        # A learnable sum of embed to document.
        if self.learnable_docu_embed:
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
        embed_token = self.bert(tokens)[0]

        if self.learnable_docu_embed:
            bsz = embed_token.shape[0]
            max_len = embed_token.shape[1]
            embed_token = embed_token.permute(0, 2, 1).reshape(-1, max_len)
            embed_docu = self.docu_embed(embed_token)
            embed_docu = embed_docu.view(bsz, -1)
        else:
            embed_docu = torch.mean(embed_token, axis=1)

        return self.g(embed_docu), self.q1(embed_docu), self.q0(embed_docu)


def true_casual_effect(data_loader, effect='ate', estimation='Q'):
    dataset = data_loader.dataset

    Q1 = dataset.treatment * dataset.response + (1 - dataset.treatment) * dataset.pseudo_response
    Q1 = Q1.cpu().data.numpy().squeeze()

    Q0 = dataset.treatment * dataset.pseudo_response + (1 - dataset.treatment) * dataset.response
    Q0 = Q0.cpu().data.numpy().squeeze()

    treatment = dataset.treatment.cpu().data.numpy().squeeze()
    prop_scores = dataset.prop_score.cpu().data.numpy().squeeze()

    if estimation == 'q':
        if effect == 'att':
            phi = (treatment * (Q1 - Q0))
            return phi.sum() / treatment.sum()
        elif effect == 'ate':
            return (Q1 - Q0).mean()

    elif estimation == 'plugin':
        phi = (prop_scores * (Q1 - Q0)).mean()
        if effect == 'att':
            return phi / treatment.mean()
        elif effect == 'ate':
            return phi


def est_casual_effect(data_loader, causal_bert, effect='ate', estimation='Q'):
    # We use `real_treatment` here to emphasize the estimations use real instead of estimated treatment.
    real_response, real_treatment = [], []
    prop_scores, Q1, Q0 = [], [], []

    causal_bert.eval()
    with torch.no_grad():
        for idx, (tokens, treatment, response) in enumerate(data_loader):
            real_response.append(response.cpu().data.numpy().squeeze())
            real_treatment.append(treatment.cpu().data.numpy().squeeze())

            prop_score, q1, q0 = causal_bert(tokens)

            prop_scores.append(prop_score.cpu().data.numpy().squeeze())
            Q1.append(q1.cpu().data.numpy().squeeze())
            Q0.append(q0.cpu().data.numpy().squeeze())

        real_response = np.concatenate(real_response, axis=0)
        real_treatment = np.concatenate(real_treatment, axis=0)

        Q1 = np.concatenate(Q1, axis=0)
        Q0 = np.concatenate(Q0, axis=0)
        prop_scores = np.concatenate(prop_scores, axis=0)

    causal_bert.train()

    if estimation == 'q':
        if effect == 'att':
            phi = (real_treatment * (Q1 - Q0))
            return phi.sum() / real_treatment.sum()
        elif effect == 'ate':
            return (Q1 - Q0).mean()

    elif estimation == 'plugin':
        phi = (prop_scores * (Q1 - Q0)).mean()
        if effect == 'att':
            return phi / real_treatment.mean()
        elif effect == 'ate':
            return phi


def causal_bert_task(args):
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

    if args.dev:
        train_group = list(range(3))
    else:
        train_group = list(range(9))

    trainset = CausalBertDataset(tokenizer=tokenizer, data_type='merged', is_unidiag=True,
                                 alpha=args.alpha, beta=args.beta, c=args.c, i=args.i,
                                 group=train_group, max_length=512, min_length=10, truncate_method='first',
                                 device=device)
    testset = CausalBertDataset(tokenizer=tokenizer, data_type='merged', is_unidiag=True,
                                 alpha=args.alpha, beta=args.beta, c=args.c, i=args.i,
                                 group=[9], max_length=512, min_length=10, truncate_method='first',
                                 device=device)

    train_loader = DataLoader(trainset, batch_size=args.bsz, drop_last=True, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.bsz, drop_last=True, shuffle=True)
    print('Finish: load data (and encode to token sequence.)')
    print('*' * 200)

    bert = BertModel.from_pretrained(trained_bert).to(device)
    causal_bert = CausalBert(bert, learnable_docu_embed=False).to(device)

    optimizer = torch.optim.Adam(causal_bert.parameters(), lr=5e-5)
    q_loss = nn.BCELoss()
    prop_score_loss = nn.BCELoss()

    print('Finish: create model.')
    real_att_q = true_casual_effect(test_loader, effect, estimation)
    est_att_q = est_casual_effect(test_loader, causal_bert, effect, estimation)
    print(f'Real: [effect: {effect}], [estimation: {estimation}], [value: {real_att_q:.5f}]')
    print(f'Before training: [effect: {effect}], [estimation: {estimation}], [value: {est_att_q:.5f}]')
    print('*' * 200)

    for e in trange(1, args.epochs + 1, desc='Epoch', disable=False):
        causal_bert.train()

        rg_loss, rq1_loss, rq0_loss = [0.] * 3
        epoch_iterator = tqdm(train_loader, desc="Iteration", disable=False)
        for step, (tokens, treatment, response) in enumerate(epoch_iterator):
            optimizer.zero_grad()
            prop_score, q1, q0 = causal_bert(tokens)

            g_loss = prop_score_loss(prop_score, treatment)
            g_loss.backward(retain_graph=True)
            rg_loss += g_loss.item()

            if len(q1[treatment == 1]) > 0:
                q1_loss = q_loss(q1[treatment == 1], response[treatment == 1])
                q1_loss.backward(retain_graph=True)
                rq1_loss += q1_loss.item()

            if len(q1[treatment == 0]) > 0:
                q0_loss = q_loss(q0[treatment == 0], response[treatment == 0])
                q0_loss.backward()
                rq0_loss += q0_loss.item()

            optimizer.step()

            epoch_iterator.set_description('Epoch {}/{}'.format(e, args.epochs))
            # epoch_iterator.set_postfix(g_loss='{:.5f}'.format(rg_loss / (step + 1)),
            #     q1_loss='{:.5f}'.format(rq1_loss / (step + 1)), q0_loss='{:.5f}'.format(rq0_loss / (step + 1)), )

            epoch_iterator.set_postfix(g_loss='{:.5f}'.format(g_loss.item()),
                q1_loss='{:.5f}'.format(q1_loss.item()), q0_loss='{:.5f}'.format(q0_loss.item()), )

        print('Finsh Epoch {}/{}'.format(e, args.epochs) + f'[effect: {effect}], [estimation: {estimation}], ' +
              # 'train_effect: {:.5f}, '.format(est_casual_effect(train_loader, causal_bert, effect, estimation)) +
            'test_effect: {:.5f}.'.format(est_casual_effect(test_loader, causal_bert, effect, estimation)))
        rg_loss, rq1_loss, rq0_loss = [0.] * 3

    print('Finish training...')

    # With only 3 groups to train.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['daily', 'merged'], default='merged')
    parser.add_argument('--diag', type=str, choices=['uni', 'raw'], default='uni',
                        help='Which tokens to use: uni for Uni-diag code, and raw for raw data.')
    # parser.add_argument('--create-vocab', action='stro, choices=['daily', 'merged'], default='merged')
    parser.add_argument('--truncate', type=str, choices=['first', 'last', 'random'], default='first')
    parser.add_argument('--min-length', type=int, default=10, help='Min length of a sequence to be used in Bert')
    parser.add_argument('--max-length', type=int, default=512, help='Max length of a sequence used in Bert')
    parser.add_argument('--bsz', type=int, default=16, help='Batch size in training')
    parser.add_argument('--epochs', type=int, default=10, help='Epoch in production version')
    parser.add_argument('--force-new', action='store_true', default=False, help='Force to train a new MLM.')
    parser.add_argument('--model', type=str, choices=['dev', 'behrt', 'med-bert'], default='behrt',
                        help='Which bert to use')

    parser.add_argument('--alpha', type=float, default=.25, help='')
    parser.add_argument('--beta', type=float, default=1, help='')
    parser.add_argument('--c', type=float, default=.2, help='')
    parser.add_argument('--i', type=float, default=0, help='')

    parser.add_argument('--effect', type=str, default='ate', choices=['ate', 'att'], help='Causal Effect')
    parser.add_argument('--estimation', type=str, default='q', choices=['q', 'plugin'], help='Estimation method')

    parser.add_argument('--cuda', type=str, help='Visible CUDA to the task.')
    parser.add_argument('--no-eval', action='store_true', default=False,
                        help='Do not evaluate during training.')

    parser.add_argument('--dev', action='store_true', help='Dev mode only use 3 groups in training.')

    args = parser.parse_args()

    effect = args.effect
    estimation = args.estimation

    args.unidiag = (args.diag == 'uni')
    args.eval_when_train = not args.no_eval
    trained_bert = '/nfs/turbo/lsa-regier/bert-results/results/behrt/MLM/merged/unidiag/checkpoint-4574003/'

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
    causal_bert_task(args)

    print('Finish all...')
