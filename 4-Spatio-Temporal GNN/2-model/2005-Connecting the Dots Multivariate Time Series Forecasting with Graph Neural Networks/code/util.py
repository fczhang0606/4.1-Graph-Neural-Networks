import torch
from torch.autograd import Variable
# 用于存储和操作张量（Tensor）的类，可以跟踪张量的计算历史，从而实现自动微分

import numpy as np
import os
import pickle
import scipy.sparse as sp  # 稀疏矩阵
from scipy.sparse import linalg  # 处理线性运算


# 对XX数据的归一化
def normal_std(x) :
    # .std()求标准差
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


class DataLoaderS(object) :

    # train and valid is the ratio of training set and validation set.
    # test = 1 - train - valid

    # exchange_rate.txt : 7588*8


    def __init__(self, file_name, train, valid, device, window, horizon, normalize=2) :

        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape  # n=7588, m=8

        self.device = device

        self.normalize = 2
        self._normalized(normalize)  # normalize模式选择，对self.dat归一化

        self.P = window   # 往回看的时间步
        self.h = horizon  # 向前看的时间步 ?= 预测的时间步（时刻/区间）
        self._split(int(train*self.n), int((train+valid)*self.n), self.n)  # 划分数据集

        self.scale = np.ones(self.m)  # 1*8
        self.scale = torch.from_numpy(self.scale).float()  # numpy转换为tensor
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)  # 拓展为指定形状的张量
        # self.test[1]是什么？.size(0)是什么？为什么要拓展？
        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)                               # rse指标
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))  # rae指标


    # normalize模式选择，对self.dat归一化
    def _normalized(self, normalize) :

        if (normalize == 0) :
            self.dat = self.rawdat

        # normalized by the maximum value of entire matrix.
        if (normalize == 1) :
            self.dat = self.rawdat / np.max(self.rawdat)

        # normalized by the maximum value of each row(sensor).
        if (normalize == 2) :
            for i in range(self.m) :  # column?
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))  # 列缩放分母
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))


    # 划分数据集
    def _split(self, train, valid, test) : # 

        train_set = range(self.P + self.h - 1, train)  # 顺序第一部分，步长为1
        valid_set = range(train, valid)  # 顺序第二部分
        test_set  = range(valid, test)   # 顺序第三部分

        # 集之中生成数据对的X和Y张量集合
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test  = self._batchify(test_set)


    # 集之中生成数据对的X和Y张量集合
    def _batchify(self, idx_set) :

        n = len(idx_set)  # 集中数据对条数
        X = torch.zeros((n, self.P, self.m))  # w步输入，输入部分的张量集合
        # Y = torch.zeros((n, self.h, self.m))
        Y = torch.zeros((n, self.m))          # 1步输出，输出部分的张量集合

        for i in range(n) :
            end   = idx_set[i] + 1 - self.h  # Y的初始位置
            start = end - self.P             # X的初始位置
            X[i, :, :] = torch.from_numpy(self.dat[start:end       , :])  # 左闭右开
            Y[i, :]    = torch.from_numpy(self.dat[end:idx_set[i]+1, :])  # end:idx_set[i]+1 ?

        return [X, Y]


    # [X, Y]中做[length/batch_size]+1多个batch
    def get_batches(self, inputs, targets, batch_size, shuffle=True) :

        length = len(inputs)

        if shuffle :
            index = torch.randperm(length)  # 随机排列的整数序列
        else :
            index = torch.LongTensor(range(length))

        start_idx = 0  # 在index中的索引号
        while (start_idx < length) :  # [length/batch_size]+1
            end_idx = min(length, start_idx + batch_size)  # 最后一个batch可能不规整
            excerpt = index[start_idx:end_idx]  # 摘录，节选
            X = inputs [excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)  # 返回？
            start_idx += batch_size


class DataLoaderM(object) :


    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True) :

        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """

        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample :
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys


    def shuffle(self) :
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys


    def get_iterator(self) :
        self.current_ind = 0

        def _wrapper() :
            while self.current_ind < self.num_batch :
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler() :

    """
    Standard the input
    """

    def __init__(self, mean, std) :
        self.mean = mean
        self.std = std

    def transform(self, data) :
        return (data - self.mean) / self.std

    def inverse_transform(self, data) :
        return (data * self.std) + self.mean


def sym_adj(adj) :

    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj) :

    """Asymmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)

    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj) :

    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """

    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True) :

    if undirected :
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)

    if lambda_max is None :
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]

    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I

    return L.astype(np.float32).todense()


def load_pickle(pickle_file) :

    try:
        with open(pickle_file, 'rb') as f :
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e :
        with open(pickle_file, 'rb') as f :
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e :
        print('Unable to load data ', pickle_file, ':', e)
        raise

    return pickle_data


def load_adj(pkl_filename) :
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None) :

    data = {}
    for category in ['train', 'val', 'test'] :
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    # Data format
    for category in ['train', 'val', 'test'] :
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader']   = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader']  = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data


def masked_mse(preds, labels, null_val=np.nan) :

    if np.isnan(null_val) :
        mask = ~torch.isnan(labels)
    else :
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan) :
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan) :

    if np.isnan(null_val) :
        mask = ~torch.isnan(labels)
    else :
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan) :

    if np.isnan(null_val) :
        mask = ~torch.isnan(labels)
    else :
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def metric(pred, real) :

    mae  = masked_mae (pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()

    return mae, mape, rmse


def load_node_feature(path) :

    fi = open(path)
    x = []
    for li in fi :
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = torch.tensor((x-mean)/std, dtype=torch.float)

    return z


def normal_std(x) :
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

