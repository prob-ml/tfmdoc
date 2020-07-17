import os
import json

from utils import VOCAB_PATH, DATA_PATH, make_dirs


def create_vocab(merged=True, group=None):
    # assert group is not None or all_group is True, "Vocab: Specify a particular user group with `group`, " \
    #                                                    "or set `all_group` to True to use all group!"

    # assert all_group is True, "Haven't support training Bert on parts of the user group data yet."

    make_dirs(VOCAB_PATH)
    vocab_file = os.path.join(VOCAB_PATH, 'vocab_merged.json' if merged else 'vocab.json')

    # Need to create vocab file.
    if not os.path.exists(vocab_file):
        user_group = [str(i) for i in range(10)] if group is None else [str(group)]

        vocab, size, maxlen = {}, 0, 0
        for group in user_group:
            read = os.path.join(DATA_PATH, f'group_{group}_merged.csv' if merged else f'group_{group}.csv')

            with open(read, 'r') as raw:
                for line in raw:
                    line = line.replace('\n', '')
                    user, hist = line.split(",")
                    hist = hist.strip()
                    tokens = hist.split(' ')

                    for token in tokens:
                        if token != '[SEP]':
                            if token in vocab:
                                # If a token is existed, don't update anything.
                                pass
                            else:
                                # A new token:
                                vocab[token] = size
                                size += 1

        for j, v in enumerate(['[UNK]', '[SEP]', '[CLS]']):
            vocab[v] = size + j

        with open(vocab_file, 'w') as vocab_write:
            json.dump(vocab, vocab_write)

    return vocab_file




