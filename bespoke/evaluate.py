import collections
from typing import List, Union, Set

import numpy as np


def compare_comm(pred_comm: Union[List, Set],
                 true_comm: Union[List, Set]) -> (float, float, float, float):
    """
    Compute the Precision, Recall, F1 and Jaccard similarity
    as the second argument is the ground truth community.
    """
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return p, r, f, j


def eval_comms_bidirectional(x_comms: List[Union[List, Set]],
                             y_comms: List[Union[List, Set]]) -> (np.ndarray, np.ndarray):
    """
    Compute the P, R, F1, Jaccard scores from both two axes.
    """
    x_node_comms = collections.defaultdict(set)
    y_node_comms = collections.defaultdict(set)
    for i, nodes in enumerate(x_comms):
        for u in nodes:
            x_node_comms[u].add(i)  # Node u is in the Community i.
    for i, nodes in enumerate(y_comms):
        for u in nodes:
            y_node_comms[u].add(i)
    # Only a small number of comparision is necessary.
    x_neighbors = collections.defaultdict(set)
    y_neighbors = collections.defaultdict(set)
    for u in x_node_comms.keys() & y_node_comms.keys():
        x_idx = x_node_comms[u]
        y_idx = y_node_comms[u]
        # x_comm_i and y_comm_j overlap for any i in x_idx and j in y_idx.
        for xid in x_idx:
            x_neighbors[xid].update(y_idx)
        for yid in y_idx:
            y_neighbors[yid].update(x_idx)
    cache = {}
    x_metrics = np.zeros([len(x_comms), 4])
    y_metrics = np.zeros([len(y_comms), 4])
    for i, neighbor_js in x_neighbors.items():
        np.max([cache.setdefault((i, j), compare_comm(x_comms[i], y_comms[j]))
                for j in neighbor_js], 0, out=x_metrics[i])
    for j, neighbor_is in y_neighbors.items():
        np.max([cache[(i, j)] for i in neighbor_is], 0, out=y_metrics[j])
    y_metrics[:, :2] = y_metrics[:, [1, 0]]  # swap p <-> r for y
    return x_metrics, y_metrics
