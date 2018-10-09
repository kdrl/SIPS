SIPS : PyTorch implementation of the paper "Graph Embedding with Shifted Inner Product Similarity and Its Improved Approximation Capability"
=======================================================

**SIPS** is an open source implementation of the paper "[Graph Embedding with Shifted Inner Product Similarity and Its Improved Approximation Capability](https://arxiv.org/abs/1810.03463)".

## Requirements & Environment

- Tested on CentOS Linux release 7.4.1708 with one GeForce GTX 1080 GPU
- Python3
- PyTorch 0.4.0
- NumPy
- scikit-learn
- tqdm

## Setup & Datasets

    .
    ├── co_author.sh
    ├── data.py
    ├── DBLP
    │   ├── column-mapping.txt
    │   ├── Conferences_Journals.txt
    │   ├── db-conferences.txt
    │   ├── dblp_node_mapping.txt
    │   ├── db_normalz_clus.txt
    │   ├── db-normalz.txt
    │   ├── graph_dblp.txt
    │   ├── node-mapping.txt
    │   └── readme.txt
    ├── main.py
    ├── models.py
    ├── README.md
    ├── train.py
    ├── wordnet
    │   ├── animal_closure.tsv
    │   ├── Makefile
    │   ├── noun_closure.tsv
    │   └── transitive_closure.py
    └── wordnet_animal.sh

### Links

- [DBLP](https://perso.liris.cnrs.fr/marc.plantevit/doku/doku.php?id=data_sets)
- [WordNet](https://github.com/facebookresearch/poincare-embeddings/tree/9aecb8634abbbc23338c2b5e5bb61ca517581bef/wordnet) : To make `animal_closure.tsv`, change `mammal` to `animal` in `Makefile`, and run `make`.

## How to run

```sh
bash wordnet_animal.sh "${DISTFN}" "${K}" "${SEED}"
bash co_author.sh "${DISTFN}" "${K}" "${SEED}"
```

### Example

```sh
> bash co_author.sh sipsnn 10 0
3.6.1 |Anaconda custom (64-bit)| (default, May 11 2017, 13:09:58)
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
co_author_0.01_sipsnn_10_0  START
Random Seed:  0
Use CUDA with  GeForce GTX 1080
Seed:0, Train:34223, Validation:3803, Test:4226
co_authorship_network: total_author_num=42252, train_author_num=34223, total_edge_num=210320, train_edge_num=138619, test_adjacency_num=4142, valid_adjacency_num=3676, train_vectors_shape=(34223, 33), vectors_shape=(42252, 33)
Indexing data
Init. features with given weight
Setting: {'name': 'co_author_0.01_sipsnn_10_0', 'exp': 'co_author', 'dsetdir': 'DBLP', 'dim': 10, 'distfn': 'sipsnn', 'lr': 0.01, 'iters': 300000, 'batchsize': 64, 'negs': 10, 'nproc': 1, 'ndproc': 2, 'neproc': 30, 'eval_each': 5000, 'debug':
True, 'undirect': True, 'seed': 0, 'hidden_layer_num': 1, 'hidden_size': 10000, 'cuda': True}
SIPSNN(
  (features): Embedding(34223, 33)
  (fc): Sequential(
    (0): Linear(in_features=33, out_features=10000, bias=True)
    (1): ReLU(inplace)
  )
  (lt): Linear(in_features=10000, out_features=9, bias=True)
  (lt_bias): Linear(in_features=10000, out_features=1, bias=True)
)
Model Parameters
Skip init. features.weight as it is given
Add fc.0.weight to trainable targets
Init. fc.0.weight with He
Add fc.0.bias to trainable targets
Init. fc.0.bias to zero
Add lt.weight to trainable targets
Init. lt.weight with He
Add lt.bias to trainable targets
Init. lt.bias to zero
Add lt_bias.weight to trainable targets
Init. lt_bias.weight to zero
Add lt_bias.bias to trainable targets
Init. lt_bias.bias to zero
Trainable parameter num :  440010
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.01
    weight_decay: 0
)
json_conf: {"distfn": "sipsnn", "dim": 10, "lr": 0.01, "batchsize": 64, "negs": 10}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 5000, "loss": 1.408626, "elapsed (for 5000 iter.)": 43.18, "val_auc": 0.889036, "best_val_auc": 0.000000}"test_auc@best_val_auc": 0.000000}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Test: {"iter": 5000, "auc": 0.883534, }
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 10000, "loss": 1.367299, "elapsed (for 5000 iter.)": 43.81, "val_auc": 0.891039, "best_val_auc": 0.889036}"test_auc@best_val_auc": 0.883534}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Test: {"iter": 10000, "auc": 0.887781, }
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 15000, "loss": 1.342593, "elapsed (for 5000 iter.)": 42.48, "val_auc": 0.892307, "best_val_auc": 0.891039}"test_auc@best_val_auc": 0.887781}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Test: {"iter": 15000, "auc": 0.889559, }
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 20000, "loss": 1.331635, "elapsed (for 5000 iter.)": 43.10, "val_auc": 0.888642, "best_val_auc": 0.892307}"test_auc@best_val_auc": 0.889559}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 25000, "loss": 1.326281, "elapsed (for 5000 iter.)": 42.29, "val_auc": 0.885577, "best_val_auc": 0.892307}"test_auc@best_val_auc": 0.889559}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 30000, "loss": 1.327210, "elapsed (for 5000 iter.)": 42.45, "val_auc": 0.893125, "best_val_auc": 0.892307}"test_auc@best_val_auc": 0.889559}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Test: {"iter": 30000, "auc": 0.892575, }
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 35000, "loss": 1.323923, "elapsed (for 5000 iter.)": 42.91, "val_auc": 0.895090, "best_val_auc": 0.893125}"test_auc@best_val_auc": 0.892575}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Test: {"iter": 35000, "auc": 0.892985, }
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 40000, "loss": 1.332860, "elapsed (for 5000 iter.)": 42.89, "val_auc": 0.893064, "best_val_auc": 0.895090}"test_auc@best_val_auc": 0.892985}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 45000, "loss": 1.321208, "elapsed (for 5000 iter.)": 42.77, "val_auc": 0.893318, "best_val_auc": 0.895090}"test_auc@best_val_auc": 0.892985}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 50000, "loss": 1.323548, "elapsed (for 5000 iter.)": 43.72, "val_auc": 0.893002, "best_val_auc": 0.895090}"test_auc@best_val_auc": 0.892985}

...

[co_author_0.01_sipsnn_10_0] Validation: {"iter": 275000, "loss": 1.326620, "elapsed (for 5000 iter.)": 43.71, "val_auc": 0.895087, "best_val_auc": 0.896559}"test_auc@best_val_auc": 0.893561}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 280000, "loss": 1.321327, "elapsed (for 5000 iter.)": 42.19, "val_auc": 0.893272, "best_val_auc": 0.896559}"test_auc@best_val_auc": 0.893561}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 285000, "loss": 1.322842, "elapsed (for 5000 iter.)": 44.27, "val_auc": 0.894554, "best_val_auc": 0.896559}"test_auc@best_val_auc": 0.893561}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 290000, "loss": 1.322403, "elapsed (for 5000 iter.)": 42.60, "val_auc": 0.892756, "best_val_auc": 0.896559}"test_auc@best_val_auc": 0.893561}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 295000, "loss": 1.333307, "elapsed (for 5000 iter.)": 43.23, "val_auc": 0.894009, "best_val_auc": 0.896559}"test_auc@best_val_auc": 0.893561}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Validation: {"iter": 300000, "loss": 1.328414, "elapsed (for 5000 iter.)": 43.27, "val_auc": 0.890375, "best_val_auc": 0.896559}"test_auc@best_val_auc": 0.893561}
Eval: 100% (00:00 left)
[co_author_0.01_sipsnn_10_0] Test@LastIteration: {"iter": 300000, "test_auc": 0.887130, "test_auc@best_val_auc": 0.893561}}
 save model

real    47m22.952s
user    371m27.355s
sys     14m17.583s
```

```sh
> bash wordnet_animal.sh sips 10 0
3.6.1 |Anaconda custom (64-bit)| (default, May 11 2017, 13:09:58)
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
wordnet_animal_0.001_sips_10_0  START
Random Seed:  0
Use CUDA with  GeForce GTX 1080
wordnet: objects_num=4027, edge_num=53905, adjacency_num=4027
Indexing data
Setting: {'name': 'wordnet_animal_0.001_sips_10_0', 'exp': 'wordnet', 'dsetdir': 'wordnet', 'dim': 10, 'distfn': 'sips', 'lr': 0.001, 'iters': 150000, 'batchsize': 128, 'negs': 20, 'nproc': 1, 'ndproc': 4, 'neproc': 10, 'eval_each': 5000, 'debug':: True, 'undirect': True, 'seed': 0, 'hidden_layer_num': 1, 'hidden_size': 1000, 'cuda': True}
SIPS(
  (lt): Embedding(4027, 9)
  (lt_bias): Embedding(4027, 1)
)
Model Parameters
Add lt.weight to trainable targets
Init. lt.weight with He
Add lt_bias.weight to trainable targets
Init. lt_bias.weight to zero
Trainable parameter num :  40270
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    weight_decay: 0
)
json_conf: {"distfn": "sips", "dim": 10, "lr": 0.001, "batchsize": 128, "negs": 20}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 5000, "loss": 2.102410, "elapsed (for 5000 iter.)": 102.55, "elapsed (for eval.)": 13.91, "auc": 0.951686, "best_auc": 0.951686}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 10000, "loss": 1.778360, "elapsed (for 5000 iter.)": 104.66, "elapsed (for eval.)": 11.15, "auc": 0.967604, "best_auc": 0.967604}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 15000, "loss": 1.722419, "elapsed (for 5000 iter.)": 98.56, "elapsed (for eval.)": 11.00, "auc": 0.973559, "best_auc": 0.973559}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 20000, "loss": 1.693375, "elapsed (for 5000 iter.)": 108.14, "elapsed (for eval.)": 10.98, "auc": 0.976297, "best_auc": 0.976297}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 25000, "loss": 1.695201, "elapsed (for 5000 iter.)": 98.81, "elapsed (for eval.)": 11.29, "auc": 0.977687, "best_auc": 0.977687}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 30000, "loss": 1.680378, "elapsed (for 5000 iter.)": 101.48, "elapsed (for eval.)": 14.00, "auc": 0.978732, "best_auc": 0.978732}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 35000, "loss": 1.677477, "elapsed (for 5000 iter.)": 102.63, "elapsed (for eval.)": 14.45, "auc": 0.979530, "best_auc": 0.979530}

...

[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 125000, "loss": 1.684119, "elapsed (for 5000 iter.)": 104.19, "elapsed (for eval.)": 11.53, "auc": 0.982512, "best_auc": 0.982512}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 130000, "loss": 1.693148, "elapsed (for 5000 iter.)": 99.68, "elapsed (for eval.)": 11.06, "auc": 0.982613, "best_auc": 0.982613}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 135000, "loss": 1.638359, "elapsed (for 5000 iter.)": 102.03, "elapsed (for eval.)": 14.45, "auc": 0.982477, "best_auc": 0.982613}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 140000, "loss": 1.719551, "elapsed (for 5000 iter.)": 99.62, "elapsed (for eval.)": 10.84, "auc": 0.982665, "best_auc": 0.982665}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 145000, "loss": 1.687383, "elapsed (for 5000 iter.)": 89.81, "elapsed (for eval.)": 11.11, "auc": 0.982281, "best_auc": 0.982665}
Eval: 100% (00:00 left)
[wordnet_animal_0.001_sips_10_0] Eval: {"iter": 150000, "loss": 1.670953, "elapsed (for 5000 iter.)": 89.39, "elapsed (for eval.)": 10.92, "auc": 0.982428, "best_auc": 0.982665}
[wordnet_animal_0.001_sips_10_0] RESULT: {"auc": 0.982665, }
 save model

real    50m48.241s
user    203m51.175s
sys     6m44.663s
```

## Acknowledgement

Our code builds upon [Facebook's poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings), see also their nice implementation of the NIPS-17 paper "Poincaré Embeddings for Learning Hierarchical Representations".

## Reference

If you find this code useful for your research, please cite the following paper in your publication:

    @article{sips,
     title = {Graph Embedding with Shifted Inner Product Similarity and Its Improved Approximation Capability},    
     author = {Akifumi Okuno and
               Geewook Kim and
               Hidetoshi Shimodaira},
     journal = {arXiv preprint arXiv:1810.03463},
     Year = {2018}
    }


## License

This code is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

[![CC-BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
