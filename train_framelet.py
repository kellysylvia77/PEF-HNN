#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from preprocessing import *
from convert_datasets_to_pygDataset import dataset_Hypergraph
from load_other_datasets import *
from H_UFG import *
from scipy.sparse.linalg import lobpcg
from scipy import sparse
from torch.nn.parameter import Parameter
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)
# function for pre-processing
# 使用切比雪夫多项式对滤波函数进行近似，得到多项式系数
def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c

# 使用滤波器的切比雪夫近似系数和图拉普拉斯矩阵  L 来构建未抽取小波变换的矩阵操作符
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0]) # # 初始的紧框架分解矩阵（单位矩阵）
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF  #表示在第 l 层和第 j 个滤波器的情况下，通过一系列线性组合（使用切比雪夫近似系数）和图拉普拉斯矩阵 L 构建的矩阵操作符。
        FD1 = d[0, l - 1]

    return d


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, r, Lev, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.crop_len = (Lev - 1) * num_nodes
        self.Lev = Lev
        self.r = r
        self.out_features = out_features
        self.residual = residual
        self.num_nodes = num_nodes
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        # self.filter = Parameter(torch.Tensor(r * Lev *self.num_nodes, 1))   #Ming 10/05/2021
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    # def forward(self, input, adj , h0 , lamda, alpha, l):
    def forward(self, input, adj, d_list, h0, lamda, alpha, l, gamma):
        adj = torch.from_numpy(adj).to(torch.float32).to(device)
        theta = math.log(lamda / l + 1)
        # hi = torch.spmm(adj, input)
        """
        Added by Ming on 12/05/2021
        """
        # Hadamard product in spectral domain
        # a = torch.arange(start=0, end=self.Lev*self.r*self.num_nodes, step=1)
        # indices = torch.stack((a,a))
        # rand_values = torch.rand(self.Lev*self.r*self.num_nodes)
        # random_filter = torch.sparse_coo_tensor(indices, rand_values, (self.Lev*self.r*self.num_nodes, self.Lev*self.r*self.num_nodes))
        # x = torch.sparse.mm(random_filter, torch.cat(d_list, dim=0))  # sparse*sparse = sparse
        # print(self.Lev, self.r, self.num_nodes)
        # print(d_list)
        x = torch.rand(self.Lev * self.r * self.num_nodes, 1).cuda() * torch.cat(d_list, dim=0).to_dense()
        #  size: [self.Lev*self.r*self.num_nodes, self.Lev*self.r*self.num_nodes]*[self.Lev*self.r*self.num_nodes, num_nodes]
        #         = [self.Lev*self.r*self.num_nodes, num_nodes]
        # random filter has shape [r * Lev * num_nodes, r * Lev * num_nodes], saved in sparse format
        # the output x has shape [r * Lev * num_nodes, num_nodes]
        # print("x.shape",x.shape)  # (5416,2708)

        # Fast Tight Frame Reconstruction
        # print(torch.cat(d_list[self.Lev - 1:], dim=1).type())
        x = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=1).float(), x[self.crop_len:, :])
        hi =gamma * torch.matmul(x, input)+(1 - gamma) * torch.spmm(adj, input)
        # ---------------------------------------
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):

    def __init__(self, nfeat, nlayers, nhidden, num_nodes, r, Lev, nclass, dropout, lamda, alpha, gamma, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, num_nodes, r, Lev, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma

    def forward(self, x, adj, d_list):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, adj, d_list, _layers[0], self.lamda, self.alpha, i + 1, self.gamma))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    # def __init__(self, runs, info=None):
    #     self.info = info
    #     self.results = [[] for _ in range(runs)]
    #
    #
    # def add_result(self, run, result):
    #     assert len(result) == 3
    #     assert run >= 0 and run < len(self.results)
    #     self.results[run].append(result)
    #     print("len(self.results[run]", len(self.results[run]))

    def __init__(self, info=None):
        self.info = info
        self.results = {}

    def add_result(self, run, result):
        if run not in self.results:
            self.results[run] = []
        assert len(result) == 3
        self.results[run].append(result)
    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            # print("self.results[0],self.results[1]", self.results[0], self.results[1])

            result = [100 * torch.tensor(self.results[r]) for r in self.results]

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]
    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
#             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(model, data, d_list, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data.x, adj, d_list)
    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

    # Also keep track of losses
    # Remark: out_raw
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out

def eval_acc(y_true, y_pred):
    # y_true = y_true.detach().cpu().numpy()  # shape: N * 1
    # y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
    # correct = (y_pred == y_true).sum().item()
    # return correct/len(y_true)
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    #     ipdb.set_trace()
    #     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    torch.cuda.current_device()
    torch.cuda._initialized = True

if __name__ == '__main__':
    
    types = 'all' # highpass, lowpass

    # param = {'datamatrix': 'cora_ca2-8', 'dataset': 'coauthor_cora', 'All_num_layers': 64, 'nhid': 512,
    #           'lr': 0.002, 'wd': 1e-3, 'Lev': 1, 'alpha': 0.1, 'gamma': 0.2, 'lamda': 0.6, 'seed': 50,
    #           'dropout': 0.7, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #           'runs_star': 0, 'runs_end': 10} # 85.54 ± 0.74
    # param = {'datamatrix': 'dblp2-13', 'dataset': 'coauthor_dblp', 'All_num_layers': 2, 'nhid': 512,
    #          'lr': 0.002, 'wd': 1e-5, 'Lev': 1, 'alpha': 0.5, 'gamma': 0.5, 'lamda': 0.1, 'seed': 1000,
    #          'dropout': 0.7, 'feature_noise': '0', 'patience': 200}  #
    # param = {'datamatrix': 'cora2-5', 'dataset': 'cora', 'All_num_layers': 4, 'nhid': 512,
    #          'lr': 0.002, 'wd': 1e-3, 'Lev': 1, 'alpha': 0.2, 'gamma': 0.2, 'lamda': 0.3, 'seed': 50,
    #          'dropout': 0.7, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #          'runs_star': 0, 'runs_end': 1}  # 81.51 ± 0.98
    param = {'datamatrix': 'citeseer2-6', 'dataset': 'citeseer', 'All_num_layers': 16, 'nhid': 512,
             'lr': 0.002, 'wd': 1e-5, 'Lev': 1, 'alpha': 0.1, 'gamma': 0.4, 'lamda': 0.5, 'seed': 500,
             'dropout': 0.5, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
             'runs_star': 0, 'runs_end': 10}  # 74.96 ± 1.77
    # param = {'datamatrix': 'house2-8', 'dataset': 'house-committees-100', 'All_num_layers': 8, 'nhid': 512,
    #          'lr': 0.003, 'wd': 2e-3, 'Lev': 1, 'alpha': 0.1, 'gamma': 0.5, 'lamda': 0.8, 'seed': 500,
    #          'dropout': 0.6, 'feature_noise': '1', 'patience': 200, 'add_self_loop': True,
    #          'runs_star': 0, 'runs_end': 1}  # 73.25 ± 1.55
    # param = {'datamatrix': 'senate2-6', 'dataset': 'senate-committees-100', 'All_num_layers': 8, 'nhid': 256,
    #          'lr': 0.002, 'wd': 1e-5, 'Lev': 1, 'alpha': 0.8, 'gamma': 0.9, 'lamda': 0.1, 'seed': 200,
    #          'dropout': 0.8, 'feature_noise': '1', 'patience': 200, 'add_self_loop': True,
    #          'runs_star': 0, 'runs_end': 1}  # 68.45 ± 6.84

    # param = {'datamatrix': 'ModelNet402-11', 'dataset': 'ModelNet40', 'All_num_layers': 4, 'nhid': 256,
    #           'lr': 0.002, 'wd': 1e-4, 'Lev': 1, 'alpha': 0.4, 'gamma': 0.4, 'lamda': 0.7, 'seed': 50,
    #           'dropout': 0.4, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #           'runs_star': 0, 'runs_end': 10}  #

    # param = {'datamatrix': 'ModelNet40_mvcnn2-8_3-1', 'dataset': 'ModelNet40_mvcnn', 'All_num_layers': 4, 'nhid': 256,
    #           'lr': 0.002, 'wd': 1e-4, 'Lev': 1, 'alpha': 0.4, 'gamma': 0.4, 'lamda': 0.7, 'seed': 50,
    #           'dropout': 0.4, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #           'runs_star': 0, 'runs_end': 10}  92.07 ± 0.44

    # param = {'datamatrix': 'ModelNet40_gvcnn2-10', 'dataset': 'ModelNet40_gvcnn', 'All_num_layers': 4, 'nhid': 256,
    #           'lr': 0.002, 'wd': 1e-4, 'Lev': 1, 'alpha': 0.4, 'gamma': 0.4, 'lamda': 0.7, 'seed': 50,
    #           'dropout': 0.4, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #           'runs_star': 0, 'runs_end': 10}  # 97.15 ± 0.28

    # param = {'datamatrix': 'ModelNet40_mgvcnn2-10', 'dataset': 'ModelNet40_mgvcnn', 'All_num_layers': 4, 'nhid': 256,
    #           'lr': 0.002, 'wd': 1e-4, 'Lev': 1, 'alpha': 0.4, 'gamma': 0.4, 'lamda': 0.7, 'seed': 50,
    #           'dropout': 0.4, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #           'runs_star': 0, 'runs_end': 10}  # 98.80 ± 0.14

    # param = {'datamatrix': 'NTU20122-9', 'dataset': 'NTU2012', 'All_num_layers': 4, 'nhid': 512,
    #          'lr': 0.002, 'wd': 1e-4, 'Lev': 1, 'alpha': 0.5, 'gamma': 0.5, 'lamda': 0.1, 'seed': 200,
    #          'dropout': 0.2, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #          'runs_star': 0, 'runs_end': 1}  # 90.04 ± 1.33

    # param = {'datamatrix': 'NTU2012_gvcnn2-6_3-1', 'dataset': 'NTU2012_gvcnn', 'All_num_layers': 4, 'nhid': 512,
    #          'lr': 0.002, 'wd': 1e-4, 'Lev': 1, 'alpha': 0.5, 'gamma': 0.5, 'lamda': 0.1, 'seed': 200,
    #          'dropout': 0.2, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #          'runs_star': 0, 'runs_end': 10}  # NTU2012_gvcnn2-6_3-1: 93.32 ± 1.04 /

    # param = {'datamatrix': 'NTU2012_mvcnn2-8', 'dataset': 'NTU2012_mvcnn', 'All_num_layers': 4, 'nhid': 512,
    #          'lr': 0.002, 'wd': 1e-4, 'Lev': 1, 'alpha': 0.5, 'gamma': 0.5, 'lamda': 0.1, 'seed': 200,
    #          'dropout': 0.2, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #          'runs_star': 0, 'runs_end': 10}  # ±

    # param = {'datamatrix': 'NTU2012_mgvcnn2-8', 'dataset': 'NTU2012_mgvcnn', 'All_num_layers': 4, 'nhid': 512,
    #          'lr': 0.002, 'wd': 1e-4, 'Lev': 1, 'alpha': 0.5, 'gamma': 0.5, 'lamda': 0.1, 'seed': 200,
    #          'dropout': 0.2, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #          'runs_star': 0, 'runs_end': 10}  # 91.37 ± 1.58

    # param = {'datamatrix': 'twitch2-11', 'dataset': 'twitch', 'All_num_layers': 4, 'nhid': 512,
    #          'lr': 0.003, 'wd': 2e-4, 'Lev': 1, 'alpha': 0.1, 'gamma': 0.1, 'lamda': 0.8, 'seed': 50,
    #          'dropout': 0.1, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #          'runs_star': 0, 'runs_end': 10}  #

    # param = {'datamatrix': 'actor2-10', 'dataset': 'actor', 'All_num_layers': 4, 'nhid': 512,
    #           'lr': 0.003, 'wd': 2e-4, 'Lev': 1, 'alpha': 0.1, 'gamma': 0.1, 'lamda': 0.8, 'seed': 50,
    #           'dropout': 0.1, 'feature_noise': '0', 'patience': 200, 'add_self_loop': True,
    #           'runs_star': 0, 'runs_end': 1}  #

    args = utils.parse_args(param)

    # # Part 1: Load data

    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom', 'actor', 'amazon', 'twitch', 'pokec',
                        'coauthor_cora', 'coauthor_dblp',
                        'cofriend_pokec', 'cocreate_twitch', 'cooccurence_actor', 'copurchasing_amazon',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed', 'senate-committees-100', 'congress-bills-100',
                        'NTU2012_mvcnn', 'NTU2012_gvcnn', 'ModelNet40_mvcnn', 'ModelNet40_gvcnn',
                        'NTU2012_mgvcnn', 'ModelNet40_mgvcnn']

    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100',
                      'house-committees-100', 'senate-committees-100', 'congress-bills-100']

    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = './data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,
                                         feature_noise=f_noise,
                                         p2raw=p2raw)
        else:
            if dname in ['cora', 'citeseer', 'pubmed', 'actor', 'amazon', 'twitch', 'pokec']:
                p2raw = './data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = './data/AllSet_all_raw_data/coauthorship/'
            # elif dname in ['cofriend_pokec']:
            #     p2raw = './data/AllSet_all_raw_data/cofriendship/'
            # elif dname in ['cocreate_twitch']:
            #     p2raw = './data/AllSet_all_raw_data/cocreate/'
            # elif dname in ['cooccurence_actor']:
            #     p2raw = './data/AllSet_all_raw_data/cooccurence/'
            # elif dname in ['copurchasing_amazon']:
            #     p2raw = './data/AllSet_all_raw_data/copurchasing/'
            # elif dname in ['yelp']:
            #     p2raw = './data/AllSet_all_raw_data/yelp/'
            # elif dname in ['actor', 'amazon', 'twitch', 'pokec']:
            #     p2raw = './data/AllSet_all_raw_data/' + dname
            else:
                p2raw = './data/AllSet_all_raw_data/'
            # print(p2raw)
            dataset = dataset_Hypergraph(name=dname, root='./pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # print(dataset)
        data = dataset.data
        num_features = dataset.num_features
        n_cls = dataset.num_classes
        if args.dname in ['yelp', 'walmart-trips', 'senate-committees', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())  # data.y存储的是labels
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])

    data = ExtractV2E(data)

    # print(data.edge_index)
    setup_seed(args.seed)
    x = data.x

    if args.add_self_loop:
        data = Add_Self_Loops(data)  # 添加自环

    # print("num_classes:", args.n_cls )
    print("x", x)
    # print(x.shape)
    print(data.edge_index)
    # print(data.edge_index.shape)
    H = ConstructH(data)
    # H = H.to(device)
    # 替换1
    num_nodes = H.shape[0]
    # print("Number of nodes: ", num_nodes)
    L, adj = compute_L(H)  # [2708,2708]
    # print("L.shape:", L.shape)
    L = sparse.coo_matrix(L, shape=(num_nodes, num_nodes))
    # print("L.shape:", L.shape)
    lobpcg_init = np.random.rand(num_nodes, 1)  # [2708,1]
    # print("lobpcg_init.shape:", lobpcg_init.shape)
    # lambda_max, _ = lobpcg(L, lobpcg_init)
    # lambda_max = lambda_max[0]

    # extract decomposition/reconstruction Masks
    FrameType = args.FrameType

    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
        RFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')

    Lev = args.Lev  # level of transform
    s = args.s  # dilation scale
    n = args.n  # n - 1 = Degree of Chebyshev Polynomial Approximation

    # J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
    # r = len(DFilters)
    r = 1

    # get matrix operators
    # d = get_operator(L, DFilters, n, s, J, Lev)  # J传入的是nan?
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    # store the matrix operators (torch sparse format) into a list: row-by-row
    # d_list = list()
    # for i in range(r):
    #     for l in range(Lev):
    #         d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))

    import scipy
    path = '/home/wangy/code/HyperUFG/FrameletMatrix/'
    # path = 'G:/华为电脑备份/0 浙江师范大学/0 博士期间论文集/合作论文6/HyperUFG/FrameletMatrix/'
    cora = scipy.sparse.load_npz(path + param['datamatrix'] + '.npz')
    # print(cora.shape)
    # print(cora[0].shape)
    if types == 'highpass':
        cora = cora[1:]
    elif types == 'lowpass':
        cora = cora[0]
    print(cora.shape)
    dcora = cora.T * cora
    d_list = [scipy_to_torch_sparse(dcora).to(device)]

    # import scipy.sparse as sp
    # tmp_coo = sp.coo_matrix(dcora)
    # values = tmp_coo.data
    # indices = np.vstack((tmp_coo.row, tmp_coo.col))
    # i = torch.LongTensor(indices)
    # v = torch.LongTensor(values)
    # d = torch.sparse_coo_tensor(i, v, tmp_coo.shape).cuda()

    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid

    # setup_seed(1000)

    split_idx_lst = []
    for run in range(10):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)
    # print(split_idx_lst[0])
    # print(split_idx_lst[1])
    if args.cuda in [0,1,2,3,4,5,6,7]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # # Part 2: Load model



    # # Part 3: Main. Training + Evaluation

    # logger = Logger(args.runs, args)
    logger = Logger(args)

    criterion = nn.NLLLoss()
    eval_func = eval_acc

    # model.train()
    # print('MODEL:', model)

    ### Training loop ###
    edge_index = data.edge_index

    n_idxs = edge_index[0, :] - edge_index[0, :].min()
    e_idxs = edge_index[1, :] - edge_index[1, :].min()
    x = data.x

    train_acc_tensor = torch.zeros((args.runs_end-args.runs_star, args.epochs))
    val_acc_tensor = torch.zeros((args.runs_end-args.runs_star, args.epochs))
    test_acc_tensor = torch.zeros((args.runs_end-args.runs_star, args.epochs))
    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_val_loss = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_val_loss is None:
                self.best_val_loss = val_loss
            elif val_loss > self.best_val_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_val_loss = val_loss
                self.counter = 0

    results, result_vals, result_tests, best_tests, mean_tests, std_tests = {}, [], [], [], [], []
    # for run in tqdm(range(args.runs_star, args.runs_end)):
    for run in range(args.runs_star, args.runs_end):
        # print(range(args.runs_star, args.runs_end))
        # print(run)
        # print(split_idx_lst)
    # for run in range(args.runs):
        run = run - args.runs_star
        setup_seed(run)
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model = GCNII(nfeat=num_features,
                      nlayers=args.All_num_layers,
                      nhidden=args.nhid,
                      num_nodes=num_nodes,  # Ming: 10/05/2021
                      r=r,
                      Lev=Lev,
                      nclass=n_cls,
                      dropout=args.dropout,
                      lamda=args.lamda,
                      alpha=args.alpha,
                      gamma=args.gamma,
                      variant=args.variant).to(device)

        # put things to device

        model = model.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        early_stopping = EarlyStopping(patience=param['patience'], min_delta=1e-4)  # 重新初始化早停机制
        best_val = float('-inf')

        model.train()
        # if run not in results:
        #     results[run] = []
        for epoch in range(args.epochs):
            #         Training part
            model.train()
            optimizer.zero_grad()
            # out接收pred
            x = x.to(device)
            out = model(x, adj, d_list)
            # print(out)
            loss = criterion(out[train_idx], data.y[train_idx])

            # optimizer.zero_grad()
            loss.backward()
            # total_loss = loss_re + loss_cls
            optimizer.step()

            #  Evaluation part
            # return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out
            result=None
            result = evaluate(model, data, d_list, split_idx, eval_func, result)
            # print(result)
            logger.add_result(run, result[:3])

            if epoch % 100 == 0:
                # 作图
                import matplotlib.pyplot as plt
                from sklearn.manifold import TSNE

                plt.rcParams["font.sans-serif"] = ["Times New Roman"]
                plt.rcParams["axes.unicode_minus"] = False
                plt.rcParams['hatch.linewidth'] = 1.5

                plt.rcParams['savefig.dpi'] = 2048  # 保存图片分辨率
                plt.rcParams['figure.dpi'] = 2048  # 分辨率

                plt.figure(figsize=(7, 7), dpi=2048)

                # font1_size = 18
                # font2_size = 22
                font3_size = 26

                z = TSNE(n_components=2).fit_transform(result[-1].detach().cpu().numpy())

                plt.xticks([])
                plt.yticks([])

                # plt.xticks(fontsize=font1_size)
                # plt.yticks(fontsize=font1_size)
                # plt.ylabel('Accuracy(%)', fontsize=font2_size)
                # plt.xticks(x, labels=labels, fontsize=font2_size)
                # plt.legend(fontsize=font3_size)
                # plt.title('PEF-HNN', fontsize=font3_size)

                plt.scatter(z[:, 0], z[:, 1], s=70, c=data.y.detach().cpu().numpy(), cmap="Set3")
                root_save = '/home/wangy/code/Visualization/Feature Visualization/'
                plt.savefig(root_save + str(run) + '_' + str(epoch) + param['dataset'] + '_' + 'PEF-HNN' + '.pdf', bbox_inches='tight')

            # results[run].append(result[:3])
            # result_test = 100 * torch.tensor(result[2])
            # result_val = 100 * torch.tensor(result[1])
            # result_tests.append(result_test)
            # result_vals.append(result_val)

            train_acc_tensor[run, epoch] = result[0]
            val_acc_tensor[run, epoch] = result[1]
            test_acc_tensor[run, epoch] = result[2]
            # smooth_loss_tensor[run, epoch] = loss_sth

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Total Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%')
            if (epoch + 1) % 100 == 0:
                torch.cuda.empty_cache()
            early_stopping(result[4])
            if early_stopping.early_stop:
                print("Early stopping triggered")

                break

        # results_new = 100 * torch.tensor(results)
        # argmax = results_new[:, 1].argmax().item()
        # best_test.append(results_new[argmax, 2].item())
        # mean_test, std_test = np.mean(best_test), np.std(best_test)

        # results_new = [100 * torch.tensor(results[r]) for r in results]
        #
        # best_results = []
        # for r in results_new:
        #     train1 = r[:, 0].max().item()
        #     valid = r[:, 1].max().item()
        #     train2 = r[r[:, 1].argmax(), 0].item()
        #     test = r[r[:, 1].argmax(), 2].item()
        #     best_results.append((train1, valid, train2, test))
        #
        # best_result = torch.tensor(best_results)
        # r = best_result[:, 3]
        # best_test.append(r)
        best_val, best_test = logger.print_statistics()
        mean_tests.append(best_test.mean().item())
        std_tests.append(best_test.std().item())
        data_frame = pd.DataFrame(
            data={'mean_tests': mean_tests, 'std_tests': std_tests}, index=range(1, len(mean_tests) + 1))
        data_frame.to_csv('/home/wangy/code/HyperUFG/results/' + str(args.dname) + '_' + str(types) + '.csv',
                          index_label='i')
        # logger.print_statistics(run)

    ### Save results ###
    best_val, best_test = logger.print_statistics()
    res_root = 'results'
    if not osp.isdir(res_root):
        os.makedirs(res_root)
    res_root = '{}/layer_{}'.format(res_root, args.All_num_layers)
    # if not osp.isdir(res_root):
    #     os.makedirs(res_root)
    # res_root = '{}/{}'.format(res_root, args.method)
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        # cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.alpha_v}_{args.alpha_e}\n'
        # cur_line += f'{args.perturb_type}_{args.perturb_prop}\n'
        cur_line = f',{best_val.mean():.3f} ± {best_val.std():.3f}\n'
        cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}\n'
        cur_line += f'\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    res_root_2 = './storage'
    if not osp.isdir(res_root_2):
        os.makedirs(res_root_2)
    filename = f'{res_root_2}/{args.dname}_{args.feature_noise}_noise.pickle'
    data = {
        'train_acc_tensor': train_acc_tensor,
        'val_acc_tensor': val_acc_tensor,
        'test_acc_tensor': test_acc_tensor,
    }
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)

    print('All done! Exit python code')



