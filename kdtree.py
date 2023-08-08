import numpy as np

def minkowski_distance_p(x, y, p = 2):
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf:
        return np.amax(np.abs(y - x), axis = -1)
    elif p == 1:
        return np.sum(np.abs(y - x), axis = -1)
    else:
        return np.sum(np.abs(y - x) ** p, axis = -1)

class KDTree(object):
    def __init__(self, data, leafsize = 10, tau = 0):
        self.data = np.asarray(data)
        self.n, self.m = np.shape(self.data)
        self.leafsize = int(leafsize)
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        self.max = np.amax(self.data, axis = 0)
        self.min = np.amin(self.data, axis = 0)
        self.tau = tau
        self.tree = self.build(np.arange(self.n), self.max, self.min)

    class node(object):
        def __lt__(self, other):
            return id(self) < id(other)

        def __gt__(self, other):
            return id(self) > id(other)

        def __le__(self, other):
            return id(self) <= id(other)

        def __ge__(self, other):
            return id(self) >= id(other)

        def __eq__(self, other):
            return id(self) == id(other)

    class leafnode(node):
        def __init__(self, idx):
            self.idx = idx
            self.children = len(idx)

    class innernode(node):
        def __init__(self, split_dim, split, small, large):
            self.split_dim = split_dim
            self.split = split
            self.small = small
            self.large = large
            self.children = small.children + large.children

    def build(self, idx, max, min):
        if len(idx) <= self.leafsize:
            return KDTree.leafnode(idx)
        else:
            data = self.data[idx]
            d = np.argmax(max-min)
            max_ele = max[d]
            min_ele = min[d]
            if max_ele == min_ele:
                return KDTree.leafnode(idx)
            data = data[:, d]

            split = (max_ele + min_ele) / 2
            small_idx = np.nonzero(data <= split)[0]
            large_idx = np.nonzero(data > split)[0]
            if len(small_idx) == 0:
                split = np.amin(data)
                small_idx = np.nonzero(data <= split)[0]
                large_idx = np.nonzero(data > split)[0]
            if len(large_idx) == 0:
                split = np.amax(data)
                small_idx = np.nonzero(data < split)[0]
                large_idx = np.nonzero(data >= split)[0]
            if len(small_idx) == 0:
                if not np.all(data == data[0]):
                    raise ValueError("Troublesome data array: %s" % data)
                split = data[0]
                small_idx = np.arange(len(data)-1)
                large_idx = np.array([len(data)-1])

            small_max = np.copy(max)
            small_max[d] = split
            large_min = np.copy(min)
            large_min[d] = split
            return KDTree.innernode(d, split, self.build(idx[small_idx], small_max, min), self.build(idx[large_idx], max, large_min))

def get_query_leaf(x, node):
    if isinstance(node, KDTree.leafnode):
        return node.idx
    else:
        if x[node.split_dim] < node.split:
            return get_query_leaf(x, node.small)
        else:
            return get_query_leaf(x, node.large)

def get_annf_offsets(queries, indices, root, tau):
    leaves = [None] * len(queries)
    offsets = [None] * len(queries)
    distances = np.full(len(queries), np.inf)
    for i in range(len(queries)):
        leaves[i] = data = get_query_leaf(queries[i], root)
        if i - 1 > 0:
            data = np.concatenate((data, leaves[i-1]))
        for j in range(len(data)):
            if np.abs(indices[i][0] - indices[data[j]][0]) > tau and np.abs(indices[i][1] - indices[data[j]][1]) > tau:
                dist = minkowski_distance_p(queries[i], queries[data[j]])
                if dist < distances[i]:
                    distances[i] = dist
                    offsets[i] = [indices[data[j]][0] - indices[i][0], indices[data[j]][1] - indices[i][1]]
    return distances, offsets    
