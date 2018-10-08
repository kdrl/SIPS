from collections import defaultdict as ddict
import numpy as np
from numpy.random import choice, randint
import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.utils.data import Dataset


eps = 1e-5


class Embedding(nn.Module):

    def __init__(self, size, dim, max_norm=None):
        super(Embedding, self).__init__()
        self.lt = nn.Embedding(
            size, dim,
            max_norm=max_norm
        )
        self.lossfn = nn.CrossEntropyLoss

    def forward(self, inputs):
        e = self.lt(inputs)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.distfn(s, o).squeeze(-1)
        return -dists

    def embed(self):
        return [list(self.lt.parameters())[0].data.cpu().numpy()]

    def loss(self, preds, targets, weight=None, size_average=True):
        lossfn = self.lossfn(size_average=size_average, weight=weight)
        return lossfn(preds, targets)


class SD(Embedding):
    """ Squared Distance """

    def __init__(self, size, dim):
        super(SD, self).__init__(size, dim)

    def distfn(self, u, v):
        return torch.sum(torch.pow(u - v, 2), dim=-1)


class IPS(Embedding):
    """ (Negative) Inner Product Similarity """

    def __init__(self, size, dim):
        super(IPS, self).__init__(size, dim)

    def distfn(self, u, v):
        return -(torch.sum(u * v, dim=-1))


class SIPS(Embedding):
    """ (Negative) Shifted Inner Product Similarity """

    def __init__(self, size, dim):
        super(SIPS, self).__init__(size, dim)
        self.lt = nn.Embedding(
            size, dim-1
        )
        self.lt_bias = nn.Embedding(
            size, 1
        )

    def distfn(self, u, ub, v, vb):
        return -(torch.sum(u * v, dim=-1) + ub.squeeze(-1) + vb.squeeze(-1))

    def forward(self, inputs):
        e = self.lt(inputs)
        eb = self.lt_bias(inputs)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        ob = eb.narrow(1, 1, e.size(1) - 1)
        sb = eb.narrow(1, 0, 1).expand_as(ob)
        dists = self.distfn(s, sb, o, ob).squeeze(-1)
        return -dists
    def embed(self):
        e = self.lt.state_dict()['weight']
        eb = self.lt_bias.state_dict()['weight']
        return [e.data.cpu().numpy(), eb.data.cpu().numpy()]


class PDF(Function):
    """ Poincare Distance Function """

    def grad(self, x, v, sqnormx, sqnormv, sqdist):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    def forward(self, u, v):
        self.save_for_backward(u, v)
        self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
        self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
        self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    def backward(self, g):
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = self.grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = self.grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv


class PD(Embedding):
    """ Poincare Distance """

    def __init__(self, size, dim):
        super(PD, self).__init__(size, dim, max_norm=1)
        self.dist = PDF

    def distfn(self, s, o):
        return self.dist()(s, o)


class EmbeddingNN(nn.Module):

    def __init__(self, features, dim, hidden_layer_num, hidden_size, max_norm=None):
        super(EmbeddingNN, self).__init__()
        self.max_norm = max_norm
        self.features = nn.Embedding(
            features.shape[0], features.shape[1]
        )
        self.features.weight.data = torch.from_numpy(features).float()
        self.features.weight.requires_grad = False
        print("Init. features with given weight")

        self.fc = self._build_nn(features.shape[1], hidden_layer_num, hidden_size)
        self.lt = nn.Linear(hidden_size, dim)

        self.lossfn = nn.CrossEntropyLoss

    def _build_nn(self, size, hidden_layer_num, hidden_size):
        block = []
        block.extend([nn.Linear(size, hidden_size), nn.ReLU(True)])
        for i in range(1, hidden_layer_num):
            block.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(True)])
        return nn.Sequential(*block)

    def forward(self, inputs):
        e = self.fc(self.features(inputs))
        e = self.lt(e)
        if self.max_norm is not None:
            n = torch.norm(e, p=2, dim=2)
            mask = (n >= 1.0)
            f = n * mask.type(n.type())
            f[f!=0] /= (1.0-eps)
            f[f==0] = 1.0
            e = e.clone()/f.unsqueeze(2)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.distfn(s, o).squeeze(-1)
        return -dists

    def embed(self, inputs):
        e = self.fc(inputs)
        e = self.lt(e)
        if self.max_norm is not None:
            n = torch.norm(e, p=2, dim=1)
            mask = (n >= 1.0)
            f = n * mask.type(n.type())
            f[f!=0] /= (1.0-eps)
            f[f==0] = 1.0
            e = e.clone()/f.unsqueeze(1)
        return [e.data.cpu().numpy()]

    def loss(self, preds, targets, weight=None, size_average=True):
        lossfn = self.lossfn(size_average=size_average, weight=weight)
        return lossfn(preds, targets)


class SDNN(EmbeddingNN):
    """ Squared Distance with NN """

    def __init__(self, features, dim, hidden_layer_num, hidden_size):
        super(SDNN, self).__init__(features, dim, hidden_layer_num, hidden_size)

    def distfn(self, u, v):
        return torch.sum(torch.pow(u - v, 2), dim=-1)


class IPSNN(EmbeddingNN):
    """ (Negative) Inner Product Similarity with NN """

    def __init__(self, features, dim, hidden_layer_num, hidden_size):
        super(IPSNN, self).__init__(features, dim, hidden_layer_num, hidden_size)

    def distfn(self, u, v):
        return -(torch.sum(u * v, dim=-1))


class SIPSNN(EmbeddingNN):
    """ (Negative) Shifted Inner Product Similarity with NN """

    def __init__(self, features, dim, hidden_layer_num, hidden_size):
        super(SIPSNN, self).__init__(features, dim-1, hidden_layer_num, hidden_size)
        self.lt_bias = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        e = self.fc(self.features(inputs))
        eb = self.lt_bias(e)
        e = self.lt(e)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        ob = eb.narrow(1, 1, e.size(1) - 1)
        sb = eb.narrow(1, 0, 1).expand_as(ob)
        dists = self.distfn(s, sb, o, ob).squeeze(-1)
        return -dists

    def embed(self, inputs):
        e = self.fc(inputs)
        return [self.lt(e).data.cpu().numpy(), self.lt_bias(e).data.cpu().numpy()]

    def distfn(self, u, ub, v, vb):
        return -(torch.sum(u * v, dim=-1) + ub.squeeze(-1) + vb.squeeze(-1))


class PDNN(EmbeddingNN):
    """ Poincare Distance with NN """

    def __init__(self, features, dim, hidden_layer_num, hidden_size):
        super(PDNN, self).__init__(features, dim, hidden_layer_num, hidden_size, max_norm=1)
        self.dist = PDF

    def distfn(self, s, o):
        return self.dist()(s, o)


class GraphDataset(Dataset):
    _ntries = 10
    _dampening = 1

    def __init__(self, idx, train_objects_num, nnegs, unigram_size=1e8):
        print('Indexing data')
        self.idx = torch.from_numpy(idx)
        self.nnegs = nnegs
        self.train_objects_num = train_objects_num
        self.max_tries = self.nnegs * self._ntries

        self._weights = ddict(lambda: ddict(int))
        self._counts = np.ones(train_objects_num, dtype=np.float)
        for i in range(self.idx.size(0)):
            t, h, w = [int(x) for x in self.idx[i]]
            self._counts[h] += w
            self._weights[t][h] += w
        self._weights = dict(self._weights)
        nents = int(np.array(list(self._weights.keys())).max() + 1)
        assert train_objects_num == nents, f'Number of train_objects do no match: {train_objects_num} != {nents}'

        if unigram_size > 0:
            c = self._counts ** self._dampening
            self.unigram_table = choice(
                train_objects_num,
                size=int(unigram_size),
                p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.size(0)

    def __getitem__(self, i):
        t, h, _ = [int(x) for x in self.idx[i]]
        negs = set()
        ntries = 0
        nnegs = self.nnegs
        while ntries < self.max_tries and len(negs) < nnegs:
            n = randint(0, self.train_objects_num)
            if n not in self._weights[t]:
                negs.add(n)
            ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
        return torch.LongTensor(ix).view(1, len(ix)), torch.zeros(1).long() # tensor(u, v, negs...), tensor(0)

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return Variable(torch.cat(inputs, 0)), Variable(torch.cat(targets, 0))

    @classmethod
    def initialize(cls, opt, train_idx, train_objects_num, train_vectors=None):
        conf = []
        data = cls(train_idx, train_objects_num, opt.negs)

        if opt.distfn == "sd":
            model = SD(data.train_objects_num, opt.dim)
        elif opt.distfn == "ips":
            model = IPS(data.train_objects_num, opt.dim)
        elif opt.distfn == "sips":
            model = SIPS(data.train_objects_num, opt.dim)
        elif opt.distfn == "pd":
            model = PD(data.train_objects_num, opt.dim)
        else:
            assert train_vectors is not None, 'no features'
            if opt.distfn == "sdnn":
                model = SDNN(train_vectors, opt.dim, opt.hidden_layer_num, opt.hidden_size)
            elif opt.distfn == "ipsnn":
                model = IPSNN(train_vectors, opt.dim, opt.hidden_layer_num, opt.hidden_size)
            elif opt.distfn == "sipsnn":
                model = SIPSNN(train_vectors, opt.dim, opt.hidden_layer_num, opt.hidden_size)
            elif opt.distfn == "pdnn":
                model = PDNN(train_vectors, opt.dim, opt.hidden_layer_num, opt.hidden_size)

        return model, data, conf
