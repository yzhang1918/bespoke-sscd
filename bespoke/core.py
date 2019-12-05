import collections
from typing import List, Set, Dict, Union

import numpy as np
import tqdm
from scipy import sparse as sp
from sklearn.cluster import KMeans


def label_nodes(neighbors: Dict[int, Set[int]], n_roles: int) -> (np.ndarray, np.ndarray):
    """
    Compute the Jaccard similarities,
    then extract percentiles as features,
    and finally label nodes by k-means.
    """
    # Compute Jaccards
    j_scores = collections.defaultdict(list)
    n_nodes = len(neighbors)
    for u in tqdm.tqdm(range(n_nodes), desc='ComputeJaccards'):
        u_neighbors = neighbors[u]
        for v in u_neighbors:
            if v <= u:
                continue
            v_neighbors = neighbors[v]
            js = len(u_neighbors & v_neighbors) / len(u_neighbors | v_neighbors)
            j_scores[u].append(js)
            j_scores[v].append(js)
    # Extract percentiles
    n_feats = 5
    features = np.zeros([n_nodes, n_feats])
    ps = np.linspace(0, 100, n_feats)
    for u in tqdm.tqdm(range(n_nodes), desc='ExtractPercentiles'):
        np.percentile(j_scores[u], ps, out=features[u])
    # Kmeans
    kmeans = KMeans(n_roles)
    labels = kmeans.fit_predict(features)
    return labels, features


def get_comm_feature(nodes: List[int], neighbors: Dict[int, Set[int]],
                     labels: np.ndarray, n_labels: int) -> np.ndarray:
    """
    Compute the distribution of edge types as the community feature."
    """
    count_mat = np.zeros([n_labels, n_labels])
    nodes = set(nodes)
    for u in nodes:
        for v in (neighbors[u] & nodes):
            if v > u:
                continue
            i, j = labels[u], labels[v]
            i, j = (i, j) if i < j else (j, i)
            count_mat[i, j] += 1
    arr = count_mat[np.triu_indices_from(count_mat)]
    n = arr.sum()
    assert np.tril(count_mat, k=-1).sum() == 0
    arr /= n + 1e-9
    return arr


def get_patterns(comms: List[List[int]], neighbors: Dict[int, Set[int]],
                 node_labels: np.ndarray, n_patterns: int) -> (np.ndarray, List[List[int]], np.ndarray):
    """
    Compute community features and use k-means to get patterns' features, size distributions, and supports.
    """
    n_labels = node_labels.max() + 1
    comm_features = [get_comm_feature(nodes, neighbors, node_labels, n_labels) for nodes in comms]
    k_means = KMeans(n_patterns)
    comm_labels = k_means.fit_predict(comm_features)
    pattern_features = k_means.cluster_centers_
    pattern_sizes = [[] for _ in range(n_patterns)]
    pattern_support = np.zeros(n_patterns)
    for i, label in enumerate(comm_labels):
        pattern_support[label] += 1
        pattern_sizes[label].append(len(comms[i]))
    return pattern_features, pattern_sizes, pattern_support


def compute_node_pattern_score(pattern_features: np.ndarray,
                               adj_mat: sp.spmatrix,
                               neighbors: Dict[int, Set[int]],
                               node_labels: np.ndarray) -> np.ndarray:
    """
    Scoring nodes based on local structures.
    """
    n_labels = node_labels.max() + 1
    n_nodes = adj_mat.shape[0]
    count_mat = np.zeros([n_labels, n_labels])
    # Local structure
    node_local_features = np.zeros((n_nodes, n_labels * (n_labels + 1) // 2))
    for u in tqdm.tqdm(range(n_nodes), desc='NodeLocalFeature'):
        count_mat.fill(0)
        for v in neighbors[u]:
            i, j = node_labels[u], node_labels[v]
            i, j = (i, j) if i < j else (j, i)
            count_mat[i, j] += 1
        arr = count_mat[np.triu_indices_from(count_mat)]
        arr /= arr.sum()
        node_local_features[u] = arr
    # First Order: Pass 1 in the paper
    node_first_order_scores = node_local_features @ pattern_features.T
    # Second Order: Pass 2 in the paper
    deg_vec = np.array(adj_mat.sum(1)).squeeze()
    node_second_order_scores = sp.diags((adj_mat @ deg_vec) ** -1) @ adj_mat @ (
            deg_vec[:, None] * node_first_order_scores)
    node_pattern_scores = node_first_order_scores + node_second_order_scores
    return node_pattern_scores


def get_seed(target_size: int, degree_seeds: Dict[int, List[int]],
             used_seeds: Set[int], eps: int = 5) -> Union[int, None]:
    """
    Find the best seed that has never be picked before.
    """
    for deg in range(target_size - 1, target_size + eps):
        sorted_seeds = degree_seeds.get(deg, [])
        if len(sorted_seeds) == 0:
            continue
        while len(sorted_seeds):
            seed = sorted_seeds.pop()
            if seed not in used_seeds:
                used_seeds.add(seed)
                return seed
    else:
        return None
