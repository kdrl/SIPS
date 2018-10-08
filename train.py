import os
import sys
import timeit
import logging
import argparse
from collections import defaultdict as ddict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.data import DataLoader
import models, train


def wordnet(model, data, optimizer, opt, log, cuda, test_adjacency):
    loader = DataLoader(
        data,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate
    )

    min_rank = (np.Inf, -1)
    max_AUC = (0, -1)
    iter_counter = 0
    former_loss = np.Inf
    t_start = timeit.default_timer()

    while True:
        train_loss = []
        loss = None

        for inputs, targets in loader:
            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.module.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            iter_counter+=1

            if iter_counter % opt.eval_each == 0:
                model.eval()
                eval_elapsed = timeit.default_timer()
                MR, AUC, ela = evaluation(model, opt.distfn, test_adjacency, opt.neproc, cuda=cuda, verbose=True)
                eval_elapsed = timeit.default_timer() - eval_elapsed
                model.train()
                if MR < min_rank[0]:
                    min_rank = (MR, iter_counter)
                if AUC > max_AUC[0]:
                    max_AUC = (AUC, iter_counter)
                log.info(
                    ('[%s] Eval: {'
                     '"iter": %d, '
                     '"loss": %.6f, '
                     '"elapsed (for %d iter.)": %.2f, '
                     '"elapsed (for eval.)": %.2f, '
                     '"auc": %.6f, '
                     '"best_auc": %.6f'
                     '}') % (
                         opt.name, iter_counter, np.mean(train_loss), opt.eval_each, timeit.default_timer() - t_start, eval_elapsed, AUC, max_AUC[0])
                )

                former_loss = np.mean(train_loss)
                train_loss = []
                t_start = timeit.default_timer()

            if iter_counter >= opt.iters:
                log.info(
                    ('[%s] RESULT: {'
                    '"auc": %.6f, '
                    '}') % (
                         opt.name, max_AUC[0])
                )
                print(""" save model """)
                torch.save({
                    'model': model.state_dict(),
                    'auc': max_AUC[0],
                    'iteration': iter_counter
                }, f'{opt.name}.pth')
                sys.exit()


def co_author(model, data, vectors, optimizer, opt, log, cuda, test_adjacency, valid_adjacency):
    loader = DataLoader(
        data,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate
    )

    max_AUC = (0, -1)
    iter_counter = 0
    final_AUC = 0
    former_loss = np.Inf
    t_start = timeit.default_timer()

    best_model = {
        'model': None,
        'iteration': 0
    }

    while True:
        train_loss = []
        loss = None

        for inputs, targets in loader:
            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.module.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            iter_counter+=1

            if iter_counter % opt.eval_each == 0:
                model.eval()
                _, AUC, ela = evaluation(model, opt.distfn, valid_adjacency, opt.neproc, vectors, cuda=cuda, verbose=True)
                model.train()
                log.info(
                    ('[%s] Validation: {'
                     '"iter": %d, '
                     '"loss": %.6f, '
                     '"elapsed (for %d iter.)": %.2f, '
                     '"val_auc": %.6f, '
                     '"best_val_auc": %.6f}'
                     '"test_auc@best_val_auc": %.6f}') % (
                         opt.name, iter_counter, np.mean(train_loss), opt.eval_each, timeit.default_timer() - t_start, AUC, max_AUC[0], final_AUC)
                )
                if AUC > max_AUC[0]:
                    max_AUC = (AUC, iter_counter)
                    model.eval()
                    _, final_AUC, ela = evaluation(model, opt.distfn, test_adjacency, opt.neproc, vectors, cuda=cuda, verbose=True)
                    model.train()
                    log.info(
                        ('[%s] Test: {'
                        '"iter": %d, '
                         '"auc": %.6f, '
                         '}') % (
                             opt.name, iter_counter, final_AUC)
                    )
                    best_model['model'] = model.state_dict()
                    best_model['iter'] = iter_counter
                former_loss = np.mean(train_loss)
                train_loss = []
                t_start = timeit.default_timer()

            if iter_counter >= opt.iters:
                model.eval()
                _, AUC, ela = evaluation(model, opt.distfn, test_adjacency, opt.neproc, vectors, cuda=cuda, verbose=True)
                model.train()
                log.info(
                    ('[%s] Test@LastIteration: {'
                    '"iter": %d, '
                     '"test_auc": %.6f, '
                     '"test_auc@best_val_auc": %.6f}'
                     '}') % (
                         opt.name, iter_counter, AUC, final_AUC)
                )
                print(""" save model """)
                torch.save(best_model, f'{opt.name}.pth')
                sys.exit()


def evaluation(model, name, adjacency, neproc, vectors=None, cuda=False, verbose=False):
    t_start = timeit.default_timer()
    adjacency = list(adjacency.items())
    chunk = int(len(adjacency)/neproc + 1)
    if vectors is not None:
        with torch.no_grad():
            vectors = Variable(torch.from_numpy(vectors).float())
            if cuda:
                vectors = vectors.cuda()
            embeds = model.module.embed(vectors)
    else:
        embeds = model.module.embed()

    queue = mp.Manager().Queue()
    processes = []
    for rank in range(neproc):
        if "sips" in name:
            p = mp.Process(
                target=eval_sips_thread,
                args=(adjacency[rank*chunk:(rank+1)*chunk], model, embeds, queue, rank==0 and verbose)
            )
        else:
            p = mp.Process(
                target=eval_thread,
                args=(adjacency[rank*chunk:(rank+1)*chunk], model, embeds, queue, rank==0 and verbose)
            )
        p.start()
        processes.append(p)

    ranks = list()
    ap_scores = list()

    for i in range(neproc):
        msg = queue.get()
        _ranks, _ap_scores = msg
        ranks += _ranks
        ap_scores += _ap_scores

    return np.mean(ranks), np.mean(ap_scores), timeit.default_timer()-t_start


def eval_thread(adjacency_thread, model, embeds, queue, verbose):
    lt = torch.from_numpy(embeds[0])
    with torch.no_grad():
        embedding = Variable(lt)
    ranks = []
    ap_scores = []
    if verbose : bar = tqdm(desc='Eval', total=len(adjacency_thread), mininterval=1, bar_format='{desc}: {percentage:3.0f}% ({remaining} left)')
    for s, s_adjacency in adjacency_thread:
        if verbose : bar.update()
        s = torch.tensor(s)
        with torch.no_grad():
            s_e = Variable(lt[s].expand_as(embedding))
        _dists = model.module.distfn(s_e, embedding).data.cpu().numpy().flatten()
        _dists[s] = 1e+12
        _labels = np.zeros(embedding.size(0))
        _dists_masked = _dists.copy()
        _ranks = []
        for o in s_adjacency:
            o = torch.tensor(o)
            _dists_masked[o] = np.Inf
            _labels[o] = 1
        """ MAP """
        _ap_scores = roc_auc_score(_labels, -_dists)
        ap_scores.append(_ap_scores)
        for o in s_adjacency:
            o = torch.tensor(o)
            d = _dists_masked.copy()
            d[o] = _dists[o]
            """ Mean rank """
            r = np.argsort(d)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
    if verbose : bar.close()
    queue.put(
        (ranks, ap_scores)
    )


def eval_sips_thread(adjacency_thread, model, embeds, queue, verbose):
    assert(len(embeds) == 2)
    lt = torch.from_numpy(embeds[0])
    ltb = torch.from_numpy(embeds[1])
    with torch.no_grad():
        embedding = Variable(lt)
        embeddingb = Variable(ltb)
    ranks = []
    ap_scores = []
    if verbose : bar = tqdm(desc='Eval', total=len(adjacency_thread), mininterval=1, bar_format='{desc}: {percentage:3.0f}% ({remaining} left)')
    for s, s_adjacency in adjacency_thread:
        s = torch.tensor(s)
        if verbose : bar.update()
        with torch.no_grad():
            s_e = Variable(lt[s].expand_as(embedding))
            s_eb = Variable(ltb[s].expand_as(embeddingb))
        _dists = model.module.distfn(s_e, s_eb, embedding, embeddingb).data.cpu().numpy().flatten()
        _dists[s] = 1e+12
        _labels = np.zeros(embedding.size(0))
        _dists_masked = _dists.copy()
        _ranks = []
        for o in s_adjacency:
            o = torch.tensor(o)
            _dists_masked[o] = np.Inf
            _labels[o] = 1
        _ap_scores = roc_auc_score(_labels, -_dists)
        ap_scores.append(_ap_scores)
        for o in s_adjacency:
            o = torch.tensor(o)
            d = _dists_masked.copy()
            d[o] = _dists[o]
            r = np.argsort(d)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
    if verbose : bar.close()
    queue.put(
        (ranks, ap_scores)
    )
