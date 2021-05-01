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
import seaborn as sns

from sklearn.metrics import roc_auc_score

from torch.distributions.binomial import Binomial
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import DataCollatorForLanguageModeling, BertForMaskedLM, BertModel
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup, AdamW

from tokens import WordLevelBertTokenizer
from vocab import create_vocab
from data import CausalBertRealDataset
from utils import DATA_PATH, make_dirs
from causal_tokens import *

TRAINED_BERT = '/nfs/turbo/lsa-regier/bert-results/results/behrt/MLM/merged/unidiag/checkpoint-6018425/'
TRAINED_CAUSAL = '/nfs/turbo/lsa-regier/bert-results/results/casualmodel/realdata/'
make_dirs(TRAINED_CAUSAL)


class CausalBOW(nn.Module):
    """
    Use a WOB embedding architecture: this is a word-of-bag model where we take the sum (or equivalently, average) of
    token input embedding without positional information.
    """
    def __init__(self, token_embed, hidden_size=256, max_length=512, binary_response=True,
                 learnable_docu_embed=False, prop_is_logit=False):
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

    def freeze_representation(self):
        for child in self.children():
            if type(child) == nn.Embedding:
                for params in child.parameters():
                    params.requires_grad = False

    def unfreeze_representation(self):
        for child in self.children():
            if type(child) == nn.Embedding:
                for params in child.parameters():
                    params.requires_grad = True


class CausalBert(nn.Module):
    """
    Use a Bert last hidden-state embedding architecture,
    see discussion here for more details: https://github.com/huggingface/transformers/issues/1950
    """
    def __init__(self, bert, hidden_size=256, max_length=512, binary_response=True, learnable_docu_embed=False,
                 prop_is_logit=False):
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

    def freeze_representation(self):
        for child in self.children():
            if type(child) == BertModel:
                for params in child.parameters():
                    params.requires_grad = False

    def unfreeze_representation(self):
        for child in self.children():
            if type(child) == BertModel:
                for params in child.parameters():
                    params.requires_grad = True

        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

        
def est_casual_effect(data_loader, model, effect='ate', estimation='q', evaluate=True, **kwargs):
    # We use `real_treatment` here to emphasize the estimations use real instead of estimated treatment.
    real_response, real_treatment, real_prop_scores = [], [], []
    prop_scores, Q1, Q0 = [], [], []
    
    if evaluate:
        best_p_loss_test = kwargs.get('best_p_loss')
        p_loss = kwargs.get('p_loss')
        q_loss = kwargs.get('q_loss')
        p_loss_test, q1_loss_test, q0_loss_test  = [], [], []
        
    model.eval()
    for idx, (tokens, treatment, response) in enumerate(data_loader):
        real_response.append(response.cpu().data.numpy().squeeze())
        real_treatment.append(treatment.cpu().data.numpy().squeeze())

        prop_score, q1, q0 = model(tokens)

        if model.prop_is_logit:
            sigmoid = nn.Sigmoid()
            prop_scores.append(sigmoid(prop_score).cpu().data.numpy().squeeze())
        else:
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

    return effect, p_loss, q1_loss, q0_loss


def show_result(train_loss_hist, test_loss_hist, est_effect, unadjust, epoch, sep_loss=True, save_path=None):
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
        lns_est_eff = ax_r.plot(np.arange(epoch), est_effect, color='coral', label='Estimate ATE')
        lns_unad_eff = ax_r.plot(np.arange(epoch), np.ones(epoch) * unadjust, color='green', ls=':', label='Unadjusted ATE')

        lns = lns_p + lns_q  + lns_est_eff + lns_unad_eff
        labs = [l.get_label() for l in lns]
        ax_r.legend(lns, labs, loc=0)
        ax.set_ylabel('Validation loss')
        ax_r.set_ylabel('ATEs')
    
    else:
        lns_l = ax.plot(np.arange(epoch), test_loss_hist_p + test_loss_hist_q1 + test_loss_hist_q0, label='Eval loss')
        ax_r = plt.twinx()
        lns_est_eff = ax_r.plot(np.arange(epoch), est_effect, color='coral', label='Estimate ATE', ls='--')
        lns_unad_eff = ax_r.plot(np.arange(epoch), np.ones(epoch) * unadjust, color='green', ls=':', label='Unadjusted ATE')

        lns = lns_l + lns_est_eff + lns_unad_eff
        labs = [l.get_label() for l in lns]
        ax_r.legend(lns, labs, loc=0)
        ax.set_ylabel('Validation loss')
        ax_r.set_ylabel('ATEs')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def load_data(pretreat_tokens, treat_tokens, response_tokens, bsz=256, train_group=[1], test_group=[9], device='cpu'):
    if isinstance(train_group, int):
        train_group = [train_group]
    
    if isinstance(test_group, int):
        test_group = [test_group]
    
    assert not set(train_group).intersection(set(test_group)), 'Error: train group and test group have overlaps...'
    
    vocab = create_vocab(merged=True, uni_diag=True)
    tokenizer = WordLevelBertTokenizer(vocab)
    
    start = time.time()
    trainset = CausalBertRealDataset(tokenizer=tokenizer, data_type='merged', is_unidiag=True,
                                     pretreat_tokens=pretreat_tokens, treat_tokens=treat_tokens,
                                     response_tokens=response_tokens, group=train_group, max_length=512, min_length=10,
                                     truncate_method='last', device=device, seed=1)

    print(f'Load training set in {(time.time() - start):.2f} sec')
    print(f'Training set: [treated: {(trainset.treatments.mean().item()):.4f}], '
          f'Training set: [response: {(trainset.responses.mean().item()):.4f}]')

    start = time.time()
    testset = CausalBertRealDataset(tokenizer=tokenizer, data_type='merged', is_unidiag=True,
                                    pretreat_tokens=pretreat_tokens, treat_tokens=treat_tokens,
                                    response_tokens=response_tokens, group=test_group, max_length=512, min_length=10,
                                    truncate_method='last', device=device, seed=1)

    print(f'Load validation set in {(time.time() - start):.2f} sec')
    print(f'Validation set: [treated: {(testset.treatments.mean().item()):.4f}], '
          f'Validation set: [response: {(testset.responses.mean().item()):.4f}]')

    train_loader = DataLoader(trainset, batch_size=bsz, drop_last=True, shuffle=True)
    test_loader = DataLoader(testset, batch_size=bsz, drop_last=True, shuffle=True)

    print('*' * 100)
    return train_loader, test_loader


def load_model(model, hidden_size, learnable_docu_embed=False, prop_is_logit=True, device='cpu'):
    model = model.lower()
    assert model in ['bow', 'bert'], f'Error: Invalid model argument: {model}, should be one of [bow, bert]...'
    
    bert = BertModel.from_pretrained(TRAINED_BERT)
        
    if model == 'bow':
        token_embed = bert.get_input_embeddings()
        model = CausalBOW(token_embed, learnable_docu_embed=learnable_docu_embed, hidden_size=hidden_size,
                          prop_is_logit=prop_is_logit)
    else:
        model = CausalBert(bert, learnable_docu_embed=learnable_docu_embed, hidden_size=hidden_size,
                           prop_is_logit=prop_is_logit)
        
    return model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bow', choices=['bow', 'bert'], help='Which Causal Model to use.')
    parser.add_argument('--learnable_docu_embed', action='store_true', default=False,
                        help='Learn the token embedding weights in document embedding')
    parser.add_argument('--freeze_representation', action='store_true', default=False,
                        help='Freeze the representation layer for token embedding')
    parser.add_argument('--pos_weight_p', type=float, default=None, help='Pos weight in propensity score loss')
    parser.add_argument('--hidden_size', type=int, default=64, help='Batch size in training')
    parser.add_argument('--bsz', type=int, default=16, help='Batch size in training')
    parser.add_argument('--epoch', type=int, default=50, help='Epoch in production version')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')

    parser.add_argument('--treat', type=str, choices=['asthma', 'flu', 'back'],
                        default=None, help='The treatment')
    parser.add_argument('--fulldata', action='store_true', default=False,
                        help='Use all data to train/evaluate model')

    parser.add_argument('--q1_loss_scale', type=float, default=1.0, help='Scale of q1 loss')
    parser.add_argument('--q0_loss_scale', type=float, default=1.0, help='Scale of q0 loss')
    parser.add_argument('--p_loss_scale', type=float, default=1.0, help='Scale of prop score loss.')

    parser.add_argument('--effect', type=str, default='ate', choices=['ate', 'att'], help='Causal Effect')
    parser.add_argument('--estimation', type=str, default='q', choices=['q', 'plugin'], help='Estimation method')
    parser.add_argument('--cuda', type=str, help='Visible CUDA to the task.')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(f'Prepare: check process on cuda: {args.cuda}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    effect = args.effect.lower()
    estimation = args.estimation.lower()
    assert effect in ['att', 'ate'], f'Wrong effect: {effect}...'
    assert estimation in ['q', 'plugin'], f'Wrong estimation: {estimation}...'

    result_path = os.path.join(curPath, 'causal-effect', 'results', 'realdata')
    make_dirs(result_path)

    if args.treat.lower() == 'asthma':
        pretreat_tokens = None
        treat_tokens = ASTHMA_TOKENS
        response_tokens = FLU_TOKENS
        args.response = 'flu'
    elif args.treat.lower() == 'flu':
        pretreat_tokens = None
        treat_tokens = FLU_TOKENS
        response_tokens = ASTHMA_TOKENS
        args.response = 'asthma'
    elif args.treat.lower() == 'back':
        pretreat_tokens = BACKPAIN_DIAG_TOKENS
        treat_tokens = BACKPAIN_PROC_TOKENS
        response_tokens = PAINKILLER_TOKENS
        args.response = 'painkiller'

    print(f'Start: create dataset: [treatment: {args.treat}], [response: {args.response}]...')

    if args.model == 'bow' and args.bsz < 256:
        print(f'Suggestion: current bsz is {args.bsz}, which may be too small, we change it to 256.')
        args.bsz = 256

    train_group = [1]
    test_group = [9]

    if args.fulldata:
        train_group = [0, 1, 2, 3, 4]
        test_group = [5, 6, 7, 8, 9]
    train_loader, test_loader = load_data(pretreat_tokens, treat_tokens, response_tokens, args.bsz, device=device,
                                          train_group=train_group, test_group=test_group)

    unadjust = (test_loader.dataset.responses[test_loader.dataset.treatments == 1].mean() -
                test_loader.dataset.responses[test_loader.dataset.treatments == 0].mean()).item()
    print(f'Unadjusted: [value: {unadjust:.4f}]')

    model = load_model(model=args.model, hidden_size=args.hidden_size,
                       learnable_docu_embed=args.learnable_docu_embed,
                       device=device)
    if args.freeze_representation:
        print(f'Warning: frozen model may perform poorly.')
        model.freeze_representation()

    epoch_iter = len(train_loader)
    total_steps = args.epoch * epoch_iter

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if args.model == 'bert' and args.lr >= 5e-6:
        print(f'Suggestion: current lr for causal-Bert is {args.lr}, which may be too large, we change it to 1e-6.')
        args.lr = 1e-6
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-7)

    if args.pos_weight_p:
        pos_weight = args.pos_weight_p * torch.ones(1, device=device)
    else:
        pos_portion = train_loader.dataset.treatments.mean()
        pos_weight = (1 - pos_portion) / pos_portion
    prop_score_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    q_loss = nn.BCELoss()

    train_loss_hist_p, train_loss_hist_q1, train_loss_hist_q0 = [], [], []
    test_loss_hist_p, test_loss_hist_q1, test_loss_hist_q0 = [], [], []
    est_effect, prop_score_hist = [], []

    best_p_loss_test = np.inf
    best_p_loss_test_epoch = 0.

    print('Start training...')
    for e in range(1, args.epoch + 1):
        model.train()
        start = time.time()

        p_loss_scale = args.p_loss_scale

        q1_loss_scale = args.q1_loss_scale
        q0_loss_scale = args.q0_loss_scale

        # At first TWO epochs, mainly focus on the propensity score learning.
        if e == 1:
            q1_loss_scale *= 0.01
            q0_loss_scale *= 0.01

        if e == 2:
            q1_loss_scale *= 0.1
            q0_loss_scale *= 0.1

        p_loss_train, q1_loss_train, q0_loss_train = [], [], []
        for idx, (tokens, treatment, response) in enumerate(train_loader):
            optimizer.zero_grad()
            prop_score, q1, q0 = model(tokens)

            loss_p = prop_score_loss(prop_score, treatment)
            loss = loss_p * p_loss_scale
            p_loss_train.append(loss_p.item())

            if len(response[treatment == 1]) > 0:
                loss_q1 = q_loss(q1[treatment==1], response[treatment==1])
                loss += loss_q1 * q1_loss_scale
                q1_loss_train.append(loss_q1.item())

            if len(response[treatment == 0]) > 0:
                loss_q0 = q_loss(q0[treatment==0], response[treatment==0])
                loss += loss_q0 * q0_loss_scale
                q0_loss_train.append(loss_q0.item())

            loss.backward()
            optimizer.step()

        run_idx = idx

        # Evaluation.
        p_loss_train = np.array(p_loss_train).mean()
        q1_loss_train = np.array(q1_loss_train).mean()
        q0_loss_train = np.array(q0_loss_train).mean()

        print('tmp: eval on train set')
        train_effect, _, _, _ = est_casual_effect(train_loader, model, effect, estimation, evaluate=False)

        print('tmp: eval on valid set')
        test_effect, p_loss_test, q1_loss_test, q0_loss_test = \
            est_casual_effect(test_loader, model, effect, estimation, evaluate=True,
                              p_loss=prop_score_loss, q_loss=q_loss, best_p_loss=best_p_loss_test)
        print('tmp: finish eval on valid set')

        if p_loss_test < best_p_loss_test:
            best_p_loss_test = p_loss_test
            best_p_loss_test_epoch = e
            model_path = f'[BEST]_[{args.treat}-{args.response}]_C-{args.model.upper()}_{args.hidden_size}_{args.epoch}'
            if args.fulldata:
                model_path = '[FULL]_' + model_path
            model_path = model_path.replace('.', ',') + '.pth'
            model_path = os.path.join(TRAINED_CAUSAL, model_path)

            model.eval()
            torch.save(model.to('cpu'), model_path)
            model = model.to(device)
            model.train()

        train_loss_hist_p.append(p_loss_train)
        train_loss_hist_q1.append(q1_loss_train)
        train_loss_hist_q0.append(q0_loss_train)

        test_loss_hist_p.append(p_loss_test)
        test_loss_hist_q1.append(q1_loss_test)
        test_loss_hist_q0.append(q0_loss_test)

        est_effect.append(test_effect)

        print(f'''Finish: epoch: {e} / {args.epoch}, time cost: {(time.time() - start):.2f} sec, 
              Loss: [Train: p = {p_loss_train:.5f}, q = {(q1_loss_train + q0_loss_train):.5f}], 
              Loss: [Test: p = {p_loss_test:.5f}, q = {(q1_loss_test + q0_loss_test):.5f},
              Effect: [{effect}-{estimation}], [train: {train_effect:.5f}], [test: {test_effect:.5f}]''')
        print('*' * 80)
        start = time.time()

    print('Finish training...')

    train_loss_hist = dict(p=train_loss_hist_p, q1=train_loss_hist_q1, q0=train_loss_hist_q1)
    test_loss_hist = dict(p=test_loss_hist_p, q1=test_loss_hist_q1, q0=test_loss_hist_q0)

    save_path = f'[Log]_[{args.treat}-{args.response}]_C-{args.model.upper()}_{args.hidden_size}_{args.epoch}'
    save_path = save_path.replace('.', ',') + '.jpg'

    save_path = os.path.join(result_path, save_path)
    show_result(train_loss_hist, test_loss_hist, est_effect, unadjust, args.epoch, save_path=save_path)

    model_path = f'[FINAL]_[{args.treat}-{args.response}]_C-{args.model.upper()}_{args.hidden_size}_{args.epoch}'
    if args.fulldata:
        model_path = '[FULL]_' + model_path
    model_path = model_path.replace('.', ',') + '.pth'
    model_path = os.path.join(TRAINED_CAUSAL, model_path)

    model.eval()
    torch.save(model.to('cpu'), model_path)

    print(f'''Save model successfully to: 
            {model_path}...
            Finish all...''')

    test_loss_hist_p = np.array(test_loss_hist_p)
    test_loss_hist_q1 = np.array(test_loss_hist_q1)
    test_loss_hist_q0 = np.array(test_loss_hist_q0)
    test_loss_hist_q = test_loss_hist_q0 + test_loss_hist_q1
    test_loss_hist_ = test_loss_hist_q + test_loss_hist_p
    est_effect = np.array(est_effect)
    dat = np.array([test_loss_hist_p, test_loss_hist_q, test_loss_hist_, est_effect]).T
    dat = pd.DataFrame(dat, columns=['propensity_score_loss', 'Q_loss', 'total_loss', 'est EFFECT'])

    csv_path = f'[log]_[{args.treat}->{args.response}].csv'
    if args.fulldata:
        csv_path = '[FULL]_' + csv_path
    dat.to_csv(os.path.join(result_path, csv_path))

    print(f'Finish all...')