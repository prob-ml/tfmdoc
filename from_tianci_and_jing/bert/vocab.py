import os
import json

from utils import VOCAB_PATH, DATA_PATH, make_dirs


def create_vocab(merged=True, uni_diag=True, group=None):
    # assert group is not None or all_group is True, "Vocab: Specify a particular user group with `group`, " \
    #                                                    "or set `all_group` to True to use all group!"

    # assert all_group is True, "Haven't support training Bert on parts of the user group data yet."

    make_dirs(VOCAB_PATH)
    file_name = 'vocab_merged' if merged else 'vocab'
    if uni_diag:
        file_name += '_unidiag'

    vocab_file = os.path.join(VOCAB_PATH, file_name + '.json' )

    # Need to create vocab file.
    if not os.path.exists(vocab_file):
        user_group = [str(i) for i in range(10)] if group is None else [str(group)]

        vocab, size = {}, 0
        for group in user_group:
            file_name = f'group_{group}_merged' if merged else f'group_{group}'
            if uni_diag:
                file_name += '_unidiag'
            read = os.path.join(DATA_PATH, file_name + '.csv' )

            with open(read, 'r') as raw:
                for line in raw:
                    line = line.replace('\n', '')
                    user, tokens = line.split(',')
                    tokens = tokens.strip()
                    token_list = tokens.split(' ')

                    for token in token_list:
                        if token not in ['[SEP]', 'document', '']:
                            if token in vocab:
                                # If a token is existed, don't do anything.
                                pass
                            else:
                                # A new token: tokens value will start from 0
                                vocab[token] = size
                                size += 1

        for j, v in enumerate(['[UNK]', '[SEP]', '[CLS]']):
            vocab[v] = size + j

        with open(vocab_file, 'w') as vocab_write:
            json.dump(vocab, vocab_write)

    return vocab_file



