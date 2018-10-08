import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn.init as init
from torch.optim import Adam
import models, train
from data import preprocess_wordnet, preprocess_co_author_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', help='Name', type=str)
    parser.add_argument('-exp', help='Exp type: "wordnet" for WordNet or "co_author" for Co-authorship network', type=str)
    parser.add_argument('-dsetdir', help='Filepath', type=str)
    parser.add_argument('-dim', help='Embedding dim.', type=int, default=5)
    parser.add_argument('-distfn', help='Distance function', type=str)
    parser.add_argument('-lr', help='Learning rate', type=float, default=1.0)
    parser.add_argument('-iters', help='Number of iters', type=int, default=1000)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=64)
    parser.add_argument('-negs', help='Number of negative samples', type=int, default=10)
    parser.add_argument('-nproc', help='Number of processes', type=int, default=1)
    parser.add_argument('-ndproc', help='Number of data loading processes', type=int, default=4)
    parser.add_argument('-neproc', help='Number of eval processes', type=int, default=32)
    parser.add_argument('-eval_each', help='Run evaluation each n-th iter', type=int, default=100)
    parser.add_argument('-debug', help='Print debug output', action='store_true', default=True)
    parser.add_argument('-undirect', help='True if an edge is undirected', action='store_true')
    parser.add_argument('-seed', help='Random seed', type=int)
    parser.add_argument('-hidden_layer_num', help='Number of the hidden layer of networks', type=int, default=1)
    parser.add_argument('-hidden_size', help='Size of the hidden layer of networks', type=int, default=1000)
    opt = parser.parse_args()

    print(opt.name," START")
    if opt.seed == None:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    opt.cuda = torch.cuda.is_available()
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        print("Use CUDA with ", torch.cuda.get_device_name(0))

    torch.set_default_tensor_type('torch.FloatTensor')
    if opt.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    log = logging.getLogger(opt.name)
    fileHandler = logging.FileHandler(f'{opt.name}.log')
    streamHandler = logging.StreamHandler()
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.setLevel(log_level)

    vectors = None
    train_idx = None
    test_idx = None
    if opt.exp == "wordnet":
        idx, train_objects_num, test_adjacency, _ = preprocess_wordnet(opt.dsetdir, opt.undirect)
        model, data, conf = models.GraphDataset.initialize(opt, idx, train_objects_num)
    elif opt.exp == "co_author":
        train_idx, train_objects_num, valid_adjacency, test_adjacency, train_vectors, vectors = preprocess_co_author_network(opt.dsetdir, opt.undirect, opt.seed)
        model, data, conf = models.GraphDataset.initialize(opt, train_idx, train_objects_num, train_vectors)
    else:
        raise Exception("no such exp.")

    print(f'Setting: {str(opt.__dict__)}')
    print(model)

    # weight initialization and filtering
    filtered_parameters = []
    print("Model Parameters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Add {} to trainable targets".format(name))
            filtered_parameters.append(param)
        if 'features' in name:
            print("Skip init. {} since it is already given".format(name))
            continue
        if 'bias' in name:
            print("Init. {} to zero".format(name))
            init.constant_(param, 0.0)
        elif 'weight' in name:
            print("Init. {} with He".format(name))
            init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        else:
            raise Exception(name)

    params_num = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    print("Trainable parameter num : ",params_num)

    if opt.cuda:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = Adam(
        filtered_parameters,
        lr=opt.lr,
    )
    print(optimizer)

    conf = [
        ('distfn', '"{:s}"'),
        ('dim', '{:d}'),
        ('lr', '{:g}'),
        ('batchsize', '{:d}'),
        ('negs', '{:d}'),
    ] + conf
    conf = ', '.join(['"{}": {}'.format(k, f).format(getattr(opt, k)) for k, f in conf])
    log.info(f'json_conf: {{{conf}}}')

    # train
    if opt.exp == "wordnet":
        train.wordnet(model, data, optimizer, opt, log, opt.cuda, test_adjacency)
    elif opt.exp == "co_author":
        train.co_author(model, data, vectors, optimizer, opt, log, opt.cuda, test_adjacency, valid_adjacency)


if __name__ == '__main__':
    print(sys.version)
    main()
