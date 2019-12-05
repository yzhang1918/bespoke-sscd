from .core import *


class Bespoke:

    def __init__(self, n_roles=4, n_patterns=5, eps=5, unique=True):
        self.n_roles = n_roles
        self.n_patterns = n_patterns
        self.eps = eps
        self.unique = unique
        self.adj_mat = None
        self.n_nodes = None
        self.node_neighbors = None
        self.pattern_sizes = None
        self.pattern_p = None
        self.used_seeds = None
        self.node_pattern_scores = None
        self.pattern_degree_seeds = None

    def fit(self, adj_mat, train_comms):
        self.adj_mat = adj_mat
        self.n_nodes = adj_mat.shape[0]
        self.node_neighbors = {u: set(adj_mat[u].indices) for u in range(self.n_nodes)}
        # Label Nodes
        node_labels, _ = label_nodes(self.node_neighbors, self.n_roles)
        # Extract Patterns
        pattern_features, self.pattern_sizes, pattern_support = get_patterns(
            train_comms, self.node_neighbors, node_labels, self.n_patterns)
        self.pattern_p = pattern_support / pattern_support.sum()
        self.node_pattern_scores = compute_node_pattern_score(pattern_features, self.adj_mat, self.node_neighbors,
                                                              node_labels)
        self.reset_seeds()

    def reset_seeds(self):
        self.used_seeds = set()
        node_degrees = np.array(self.adj_mat.sum(1)).squeeze().astype(int)
        degree_node_dict = collections.defaultdict(list)
        for i, d in enumerate(node_degrees):
            degree_node_dict[d].append(i)

        self.pattern_degree_seeds = [{d: sorted(nodes, key=lambda i: -x[i])
                                      for d, nodes in degree_node_dict.items()}
                                     for x in self.node_pattern_scores.T]

    def sample(self):
        n_try = 0
        while (n_try < 20) and (len(self.used_seeds) < self.n_nodes):
            n_try += 1
            pattern_id = np.random.choice(len(self.pattern_p), p=self.pattern_p)
            target_size = np.random.choice(self.pattern_sizes[pattern_id])
            seed = get_seed(target_size, self.pattern_degree_seeds[pattern_id],
                            self.used_seeds if self.unique else set())
            if seed is None:
                continue
            return [seed] + list(self.node_neighbors[seed]), pattern_id, target_size
        else:
            raise ValueError('(Almost) Run out of seeds!')

    def sample_batch(self, n, reset=False):
        if reset:
            self.reset_seeds()
        pred_comms = []
        try:
            for _ in range(n):
                pred_comms.append(self.sample()[0])
        except ValueError as e:
            print('Warning!!!', e)
        return pred_comms
