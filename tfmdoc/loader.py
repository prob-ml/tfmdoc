import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler

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
        elif mode == "pretraining":
            ys = pad_sequence(ys, batch_first=True, padding_value=0)
    ws = torch.stack(ws)
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


def balanced_sampler(ix, labels):
    # given a very imbalanced data set
    # downsample the majority class and upsample the minority class
    # this will result in an approximate 50-50 split per batch
    p = labels[ix].sum() / len(ix)
    weights = 1.0 / torch.tensor([1 - p, p], dtype=torch.float)
    sample_weights = weights[labels[ix]]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def build_loaders(
    dataset,
    lengths,
    pad,
    batch_size,
    save_test_index,
    random_seed,
    mode,
):
    # create dataloaders for training, test, and (optionally)
    # check whether to create validation set
    if lengths[1] > 0:
        keys = ("train", "val", "test")
    else:
        keys = ("train", "test")
        lengths = (lengths[0], lengths[2])
    subsets = random_split(dataset, lengths, torch.Generator().manual_seed(random_seed))
    loaders = {}

    collate_fn = lambda x: padded_collate(x, pad, mode)

    for key, subset in zip(keys, subsets):
        if key == "test" or mode != "diagnosis":
            sampler = None
            # save index of test samples for post hoc analysis
            if save_test_index:
                torch.save(torch.tensor(subset.indices), "test_index.pt")
        else:
            sampler = balanced_sampler(subset.indices, dataset.labels)
        loaders[key] = DataLoader(
            subset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            sampler=sampler,
        )
    return loaders
