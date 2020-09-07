import os
import sys
import time
import argparse
import random

from tqdm.auto import tqdm, trange

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

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


TRAINED_BERT = '/nfs/turbo/lsa-regier/bert-results/results/behrt/MLM/merged/unidiag/checkpoint-6018425/'


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
    def __init__(self, bert, hidden_size=256, max_length=512, binary_response=True, learnable_docu_embed=False, prop_is_logit=False):
        super().__init__()
        self.bert = bert
        self.learnable_docu_embed = learnable_docu_embed

        embed_out = self.bert.get_input_embeddings().weight.shape[1]

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
#                   nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), nn.Sigmoid()])
#             self.q0 = nn.Sequential(
#                 *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
#                   nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), nn.Sigmoid()])
#         else:
#             self.q1 = nn.Sequential(
#                 *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
#                   nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), ])
#             self.q0 = nn.Sequential(
#                 *[nn.Linear(embed_out, hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size),
#                   nn.ReLU(inplace=True), nn.Linear(hidden_size, 1), ])

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


def true_casual_effect(data_loader, effect='ate', estimation='q'):
    assert effect == 'ate' and estimation == 'q', f'unallowed effect/estimation: {effect}/{estimation}'
    
    dataset = data_loader.dataset
    
    Q1 = dataset.treatment * dataset.response + (1 - dataset.treatment) * dataset.pseudo_response
    Q1 = Q1.cpu().data.numpy().squeeze()

    Q0 = dataset.treatment * dataset.pseudo_response + (1 - dataset.treatment) * dataset.response
    Q0 = Q0.cpu().data.numpy().squeeze()

    treatment = dataset.treatment.cpu().data.numpy().squeeze()
    prop_scores = dataset.prop_scores.cpu().data.numpy().squeeze()
    
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

        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

        
def est_casual_effect(data_loader, model, effect='ate', estimation='q', evaluate=True, **kwargs):
    # We use `real_treatment` here to emphasize the estimations use real instead of estimated treatment.
    real_response, real_treatment, real_prop_scores = [], [], []
    prop_scores, Q1, Q0 = [], [], []
    
    if evaluate:
        p_loss = kwargs.get('p_loss')
        q_loss = kwargs.get('q_loss')
        p_loss_test, q1_loss_test, q0_loss_test  = [], [], []
        
    model.eval()
    for idx, (tokens, treatment, response, real_prop_score) in enumerate(data_loader):
        real_response.append(response.cpu().data.numpy().squeeze())
        real_treatment.append(treatment.cpu().data.numpy().squeeze())
        real_prop_scores.append(real_prop_score.cpu().data.numpy().squeeze())

        prop_score, q1, q0 = model(tokens)
        
        prop_scores.append(prop_score.cpu().data.numpy().squeeze())
        Q1.append(q1.cpu().data.numpy().squeeze())
        Q0.append(q0.cpu().data.numpy().squeeze())
        
        # Evaulate loss
        if evaluate:
            p_loss_val  = p_loss(prop_score, treatment)
            p_loss_test.append(p_loss_val.item())
            
            if len(response[treatment == 1]) > 0:
                q1_loss_val = q_loss(q1[treatment==1], response[treatment==1])
                q1_loss_test.append(q1_loss_val.item())

            if len(response[treatment == 0]) > 0:
                q0_loss_val = q_loss(q0[treatment==0], response[treatment==0])
                q0_loss_test.append(q0_loss_val.item())            
               
    p_loss = np.array(p_loss_test).mean() if evaluate else None
    q1_loss = np.array(q1_loss_test).mean() if evaluate else None
    q0_loss = np.array(q0_loss_test).mean() if evaluate else None

    Q1 = np.concatenate(Q1, axis=0)
    Q0 = np.concatenate(Q0, axis=0)
    prop_scores = np.concatenate(prop_scores, axis=0)
    
    real_response = np.concatenate(real_response, axis=0)
    real_treatment = np.concatenate(real_treatment, axis=0)
    real_prop_scores = np.concatenate(real_prop_scores, axis=0)
    
    # Evaluate accuracy.
    if evaluate:
        dataset = data_loader.dataset
        
        real_q1_prob = sigmoid(dataset.alpha + dataset.beta * (real_prop_scores - dataset.c) + dataset.i)
        real_q0_prob = sigmoid(dataset.beta * (real_prop_scores - dataset.c) + dataset.i)
        thre = (real_q1_prob + real_q0_prob) / 2

    # prop score: real and estimated must locate one the same side of 0.5.
    prop_accu = (1. * (((real_prop_scores - .5) * (prop_scores - .5)) > 0.)).mean() if evaluate else None
    # q: estimate is more close to corresponding real value than the other.
    q1_accu = (1. * (dataset.alpha > 0) * (Q1 > thre)).mean() if evaluate else None
    q0_accu = (1. * (dataset.alpha > 0) * (Q0 < thre)).mean() if evaluate else None

    if estimation == 'q':
        if effect == 'att':
            phi = (real_treatment * (Q1 - Q0))
            effect = phi.sum() / real_treatment.sum()
        elif effect == 'ate':
            effect = (Q1 - Q0).mean()

    elif estimation == 'plugin':
        phi = (prop_scores * (Q1 - Q0)).mean()
        if effect == 'att':
            effect = phi / real_treatment.mean()
        elif effect == 'ate':
            effect = phi
    
    model.train()

    return effect, p_loss, q1_loss, q0_loss, prop_accu, q1_accu, q0_accu


def show_result(train_loss_hist, test_loss_hist, est_effect, real, unadjust, epoch, sep_loss=True):
    train_loss_hist_p = np.array(train_loss_hist['p'])
    train_loss_hist_q1 = np.array(train_loss_hist['q1'])
    train_loss_hist_q0 = np.array(train_loss_hist['q0'])

    test_loss_hist_p = np.array(test_loss_hist['p'])
    test_loss_hist_q1 = np.array(test_loss_hist['q1'])
    test_loss_hist_q0 = np.array(test_loss_hist['q0'])
    
    est_effect = np.array(est_effect)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if sep_loss:
        lns_p = ax.plot(np.arange(epoch), test_loss_hist_p, label='Eval loss: prop_score', color='darkgreen')
        lns_q = ax.plot(np.arange(epoch), test_loss_hist_q1 + test_loss_hist_q0,  label='Eval loss: q1+q0', color='blue')
        ax_r = plt.twinx()
        lns_est_eff = ax_r.plot(np.arange(epoch), est_effect, color='coral', label='Estimate ATE', ls='--')
        lns_real_eff = ax_r.plot(np.arange(epoch), np.ones(epoch) * real, color='red', ls=':', label='Real ATE')
        lns_unad_eff = ax_r.plot(np.arange(epoch), np.ones(epoch) * unadjust, color='green', ls=':', label='Unadjusted ATE')

        lns = lns_p + lns_q + lns_est_eff + lns_real_eff + lns_unad_eff
        labs = [l.get_label() for l in lns]
        ax_r.legend(lns, labs, loc=0)
        ax.set_ylabel('Eval loss')
        ax_r.set_ylabel('ATEs')
    
    else:
        lns_l = ax.plot(np.arange(epoch), test_loss_hist_p + test_loss_hist_q1 + test_loss_hist_q0, label='Eval loss')
        ax_r = plt.twinx()
        lns_est_eff = ax_r.plot(np.arange(epoch), est_effect, color='coral', label='Estimate ATE', ls='--')
        lns_real_eff = ax_r.plot(np.arange(epoch), np.ones(epoch) * real, color='red', ls=':', label='Real ATE')
        lns_unad_eff = ax_r.plot(np.arange(epoch), np.ones(epoch) * unadjust, color='green', ls=':', label='Unadjusted ATE')

        lns = lns_l + lns_est_eff + lns_real_eff + lns_unad_eff
        labs = [l.get_label() for l in lns]
        ax_r.legend(lns, labs, loc=0)
        ax.set_ylabel('Eval loss')
        ax_r.set_ylabel('ATEs')
        
    plt.show()
    
    

def load_data(alpha, beta, c=0.2, i=0, bsz=256, train_group=[1], test_group=[9], device='cpu'):
    if isinstance(train_group, int):
        train_group = [train_group]
    
    if isinstance(test_group, int):
        test_group = [test_group]
    
    assert not set(train_group).intersection(set(test_group)), 'Error: train group and test group have overlaps...'
    
    vocab = create_vocab(merged=True, uni_diag=True)
    tokenizer = WordLevelBertTokenizer(vocab)
    
    start = time.time()
    trainset = CausalBertDataset(tokenizer=tokenizer, data_type='merged', is_unidiag=True,
                                 alpha=alpha, beta=beta, c=c, i=i, 
                                 group=train_group, max_length=512, min_length=10,
                                 truncate_method='first', device=device, seed=1)

    print(f'Load training set in {(time.time() - start):.2f} sec')

    start = time.time()
    testset = CausalBertDataset(tokenizer=tokenizer, data_type='merged', is_unidiag=True,
                                alpha=alpha, beta=beta, c=c, i=i, 
                                group=[9], max_length=512, min_length=10,
                                truncate_method='first', device=device)

    print(f'Load validation set in {(time.time() - start):.2f} sec')
    
    train_loader = DataLoader(trainset, batch_size=bsz, drop_last=True, shuffle=True)
    test_loader = DataLoader(testset, batch_size=bsz, drop_last=True, shuffle=True)
    
    return train_loader, test_loader


def load_model(model, hidden_size, prop_is_logit=True, device='cpu'):
    model = model.lower()
    assert model in ['bow', 'bert'], f'Error: Invalid model argument: {model}, should be one of [bow, bert]...'
    
    bert = BertModel.from_pretrained(TRAINED_BERT)
        
    if model == 'bow':
        token_embed = bert.get_input_embeddings()
        model = CausalBOW(token_embed, learnable_docu_embed=False, hidden_size=hidden_size, prop_is_logit=prop_is_logit)
    else:
        model = CausalBert(bert, learnable_docu_embed=False, hidden_size=hidden_size, prop_is_logit=prop_is_logit)
        
    return model.to(device)


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
