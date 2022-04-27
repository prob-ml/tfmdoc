import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

# HELPERS


def padded_collate(batch, pad=True, mode="pretraining"):
    if mode == "early_detection":
        early, late = zip(*batch)
        ts, vs, ws, xs, ys = zip(*(early + late))
        ys = torch.tensor(ys)
    else:
        ts, vs, ws, xs, ys = zip(*batch)
        if mode == "diagnosis":
            ys = torch.stack(ys)
            ws = torch.stack(ws)
        elif mode == "pretraining":
            ys = pad_sequence(ys, batch_first=True, padding_value=0)
    if pad:
        ts = pad_sequence(ts, batch_first=True, padding_value=0)
        vs = pad_sequence(vs, batch_first=True, padding_value=0)
        xs = pad_sequence(xs, batch_first=True, padding_value=0)
    else:
        # can ignore visits in bag-of-words model
        ts = torch.stack(ts)
        vs = None
        xs = torch.stack(xs)
    # ages, visits, demog data, codes, labels
    return ts, vs, ws, xs, ys


def build_loaders(
    dataset,
    lengths,
    pad,
    batch_size,
    random_seed,
    mode,
):
    # create dataloaders for training, validation, and (optionally)
    # check whether to create a test set
    if lengths[2] > 0:
        keys = ("train", "val", "test")
    else:
        keys = ("train", "val")
        lengths = (lengths[0], lengths[1])
    subsets = random_split(dataset, lengths, torch.Generator().manual_seed(random_seed))
    loaders = {}

    collate_fn = lambda x: padded_collate(x, pad, mode)

    for key, subset in zip(keys, subsets):
        loaders[key] = DataLoader(
            subset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=(key == "train"),
        )
    return loaders


def calc_sizes(train_frac, val_frac, n):
    train_size = int(train_frac * n)
    if train_frac + val_frac == 1:
        val_size = n - train_size
        test_size = 0
    else:
        val_size = int(val_frac * n)
        test_size = n - train_size - val_size
    return train_size, val_size, test_size
