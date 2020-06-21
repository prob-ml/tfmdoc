import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import numpy as np
import pandas as pd

import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from WordLevelTokenizer import WordLevelBertTokenizer

def is_in(z, a=1, b=3):
    return (z >= a) * (z <= b)


xdim = 100  # number of billing codes
N = 20_000  # number of patients
T = 5  # total time steps

# model parameters
Psi = torch.arange (0, xdim * 0.01, 0.01)

# random variables
Z_true = torch.zeros ((N, T))  # latent
X = torch.zeros ((N, T))  # observed
X_onehot = torch.zeros ((N, T, xdim))
Y = torch.zeros ((N, 1))

for t in range (0, T):
    # Zit | Zi,t-1, Yi
    meanz = (0.9 * Z_true[:, t - 1]) if t != 0 else torch.zeros ((N,))
    Zt = Normal (meanz, 1)
    Z_true[:, t] = Zt.sample ()

    # Xit | Zit
    Psi_z = Z_true[:, t].view (N, 1) * Psi.view (1, xdim)
    PX = F.softmax (Psi_z, dim=1)
    Xt = Categorical (PX)
    Xit = Xt.sample ()
    X[:, t] = Xit
    X_onehot[:, t] = F.one_hot (Xit, num_classes=xdim)

for t in range (T - 2):
    Y[:, 0] += is_in (Z_true[:, t]) * is_in (Z_true[:, t + 1]) * is_in (Z_true[:, t + 2])

# truncate the Y values which are greater than 1.
Y = torch.cat ((Y, torch.ones ((N, 1))), 1).min (dim=1).values

X_ = X.cpu().data.numpy().astype(int)
Y_ = Y.cpu().data.numpy().astype(int)

np.savetxt('synthetic_X.txt', X_, delimiter=' ', fmt='%s')

# %%time
import time
import os
from pathlib import Path

from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import BertProcessing

# We build our custom tokenizer:
tokenizer = Tokenizer (BPE ())
tokenizer.normalizer = Lowercase ()
tokenizer.pre_tokenizer = WhitespaceSplit ()

# We can train this tokenizer by giving it a list of path to text files:
# trainer = trainers.BpeTrainer(vocab_size=200, special_tokens=[
#         "<s>",
#         "<pad>",
#         "</s>",
# #         " ",
#         "<unk>",
#         "<mask>",
#     ])

trainer = trainers.BpeTrainer (special_tokens=["[UNK]", '[SEP]', '[]'])

files = [str (x) for x in Path (".").glob ("**/synthetic_X.txt")]
print (files)
tokenizer.train (trainer, files)

# Add post_processor.
# tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")), # SEP
#     ("<s>", tokenizer.token_to_id("<s>")), # CLS
# )

if not os.path.exists ('Synthetic'):
    os.makedirs ('Synthetic')

# Set truncation.
tokenizer.enable_truncation (max_length=10)

# And now it is ready, we can save the vocabulary with
tokenizer.model.save ("./Synthetic")

# And simply use it
tokenizer.encode ('6 64 0 48').tokens

tokenizer = WordLevelBertTokenizer(vocab_file='./Synthetic/vocab.json', unk_token='<unk>')
