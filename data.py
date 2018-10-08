from itertools import count
from collections import defaultdict as ddict
from sklearn.model_selection import train_test_split
import numpy as np


def iter_line(fname, sep='\t', type=tuple, comment='#', return_idx=False, convert=None):
    with open(fname, 'r') as fin:
        if return_idx: index = -1
        for line in fin:
            if line[0] == comment:
                continue
            if convert is not None:
                d = [convert(i) for i in line.strip().split(sep)]
            else:
                d = line.strip().split(sep)
            out = type(d)
            if out is not None:
                if return_idx:
                    index += 1
                    yield (index, out)
                else:
                    yield out


def intmap_to_list(d):
    arr = [None for _ in range(len(d))]
    for v, i in d.items():
        arr[i] = v
    assert not any(x is None for x in arr)
    return arr


def preprocess_wordnet(dir_path, undirect=True):
    ecount = count()
    enames = ddict(ecount.__next__)
    subs = []
    adjacency = ddict(set)

    edge = dict()
    for i, j in iter_line(dir_path + "/animal_closure.tsv", sep='\t'):
        if not (i,j) in edge:
            edge[(i,j)] = 1
        else:
            edge[(i,j)] += 1
    if undirect:
        edge_keys = list(edge.keys())
        already_include_inverse = True
        for key in edge_keys:
            swapped_key = (key[1],key[0])
            if not swapped_key in edge:
                edge[swapped_key] = 1
                already_include_inverse = False
            else:
                edge[swapped_key] += 1
        if already_include_inverse:
            """ If the data already includes all inverse pair """
            for key in edge.keys():
                edge[key] = int(edge[key]/2)
    edge_list = list()
    for key, value in edge.items():
        for i in range(value):
            edge_list.append(key)

    for i, j in edge_list:
        if i == j: continue
        subs.append((enames[i], enames[j], 1))
        adjacency[enames[i]].add(enames[j])
    adjacency = dict(adjacency)
    idx = np.array(subs, dtype=np.int)
    edge_num = len(idx)
    if undirect:
        edge_num = int(edge_num/2)
    objects = intmap_to_list(dict(enames))
    objects_num = len(objects)
    print(f'wordnet: objects_num={objects_num}, edge_num={edge_num}, adjacency_num={len(adjacency)}')
    return idx, objects_num, adjacency, objects


def preprocess_co_author_network(dir_path, undirect=True, seed=0):
    """ Train 0.9*0.9 = 0.81, Valid 0.9*0.1 = 0.09, Test 0.1 """

    """ Parse all authors """
    author2id = dict()
    id = 0
    for line in open(dir_path + "/dblp_node_mapping.txt").readlines():
        author2id[int(line.split()[0])] = id
        id+=1

    """ Split """
    train_author, test_author = train_test_split(list(author2id.keys()), test_size=0.1, random_state=seed)
    train_author, valid_author = train_test_split(train_author, test_size=0.1, random_state=seed)
    train_author2id = dict()
    id = 0
    for i in train_author:
        if not i in train_author2id:
            train_author2id[i] = id
            id+=1
    print(f"Seed:{seed}, Train:{len(train_author)}, Validation:{len(valid_author)}, Test:{len(test_author)}")
    # print(f"EX) Train authors:{train_author[:5]}")
    train_author = set(train_author);valid_author = set(valid_author);test_author = set(test_author);

    train_objects_num = len(train_author2id)
    train_vectors = np.empty((train_objects_num, 33))
    vectors = np.empty((len(author2id), 33))
    selected_attributes = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 35, 37])
    for i, vec in iter_line(dir_path + "/db_normalz_clus.txt", sep=',', type=np.array, convert=float, return_idx=True):
        assert vec.shape[0] == 38
        vec = vec.astype(np.float32)[selected_attributes]
        vectors[author2id[i]] = vec
        if i in train_author:
            train_vectors[train_author2id[i]] = vec

    edge = dict()
    for i, j in iter_line(dir_path + "/graph_dblp.txt", sep='\t', type=tuple, convert=int):
        if not (i,j) in edge:
            edge[(i,j)] = 1
        else:
            edge[(i,j)] += 1
    if undirect:
        edge_keys = list(edge.keys())
        already_include_inverse = True
        for key in edge_keys:
            swapped_key = (key[1],key[0])
            if not swapped_key in edge:
                edge[swapped_key] = 1
                already_include_inverse = False
            else:
                edge[swapped_key] += 1
        if already_include_inverse:
            """ If the data already includes all inverse pair """
            for key in edge.keys():
                edge[key] = int(edge[key]/2)
    edge_list = list()
    for key, value in edge.items():
        for i in range(value):
            edge_list.append(key)
    total_edge_num = len(edge_list)
    if undirect:
        total_edge_num = int(total_edge_num/2)

    train_idx = []
    valid_adjacency = dict()
    test_adjacency = dict()
    for i, j in edge_list:
        if i == j: continue
        if i in train_author and j in train_author:
            u = train_author2id[i]
            v = train_author2id[j]
            train_idx.append((u,v,1))
        else:
            u = author2id[i]
            v = author2id[j]
            if i in test_author:
                if u in test_adjacency: test_adjacency[u].add(v)
                else: test_adjacency[u] = set([v])
            if i in valid_author and not j in test_author:
                if u in valid_adjacency: valid_adjacency[u].add(v)
                else: valid_adjacency[u] = set([v])

    train_idx=np.array(train_idx, dtype=np.int);
    edge_num = len(train_idx)
    if undirect:
        edge_num = int(edge_num/2)

    print(f'co_authorship_network: total_author_num={len(author2id)}, train_author_num={train_objects_num}, total_edge_num={total_edge_num}, train_edge_num={edge_num}, test_adjacency_num={len(test_adjacency)}, valid_adjacency_num={len(valid_adjacency)}, train_vectors_shape={train_vectors.shape}, vectors_shape={vectors.shape}')
    return train_idx, train_objects_num, valid_adjacency, test_adjacency, train_vectors, vectors
