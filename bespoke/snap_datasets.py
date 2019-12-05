import collections
import pathlib

import numpy as np
from scipy import sparse as sp


def load_dblp_or_amazon_network(root, name):
    edges = open(root / f'{name}/com-{name}.ungraph.txt').readlines()
    edges = [[int(i) for i in e.split()] for e in edges[4:]]
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    nodes = {i for x in edges for i in x}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}
    edges = np.asarray([[mapping[u], mapping[v]] for u, v in edges])
    comms = open(root / f'{name}/com-{name}.top5000.cmty.txt').readlines()
    comms = [[mapping[int(i)] for i in x.split()] for x in comms]
    return edges, comms, mapping


def load_email_network(root):
    edges = open(root / 'email/email-Eu-core.txt').read().strip()
    edges = [[int(i) for i in e.split(' ')] for e in edges.split('\n')]
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    nodes = {i for x in edges for i in x}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}
    edges = np.asarray([[mapping[u], mapping[v]] for u, v in edges])
    comm_membership = open(root / 'email/email-Eu-core-department-labels.txt').read().strip()
    comms = collections.defaultdict(list)
    for line in comm_membership.split('\n'):
        u, i = line.split(' ')
        mapped_u = mapping.get(int(u), None)
        if mapped_u is not None:
            comms[i].append(mapped_u)
    comms = list(comms.values())
    return edges, comms, mapping


def load_snap_dataset(name, root='datasets'):
    root = pathlib.Path(root)
    if name == 'dblp':
        edges, comms, mapping = load_dblp_or_amazon_network(root, name)
    elif name == 'amazon':
        edges, comms, mapping = load_dblp_or_amazon_network(root, name)
    elif name == 'email':
        edges, comms, mapping = load_email_network(root)
    else:
        raise NotImplementedError
    n_nodes = edges.max() + 1
    adj_mat = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=[n_nodes, n_nodes])
    adj_mat += adj_mat.T
    return adj_mat, comms, mapping
