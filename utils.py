import numpy as np
import random


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)


def split_comms(comms, n_train, seed, max_size=0):
    if max_size:
        comms = [x for x in comms if 3 <= len(x) <= max_size]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(comms))
    train_comms = [comms[i] for i in idx[:n_train]]
    test_comms = [comms[i] for i in idx[n_train:]]
    return train_comms, test_comms


def print_results(score, prefix='Metric'):
    p, r, f, j = score
    print(f'{prefix}: Precision:{p:.2f} Recall:{r:.2f} F1:{f:.2f} Jaccard:{j:.2f}', flush=True)


def write_comms_to_file(comms, fname):
    with open(fname, 'w') as fh:
        content = '\n'.join([' '.join([str(i) for i in comm]) for comm in comms])
        fh.write(content)


def read_comms_from_file(fname):
    with open(fname) as fh:
        content = fh.read().strip().split('\n')
        comms = [[int(i) for i in line.strip().split(' ')] for line in content]
    return comms



