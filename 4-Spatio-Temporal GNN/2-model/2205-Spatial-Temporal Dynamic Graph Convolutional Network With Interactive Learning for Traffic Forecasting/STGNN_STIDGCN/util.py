# 
import torch

import numpy as np
import os
import scipy.sparse as sp


# 
class DataLoader(object) :


    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True) :

        self.batch_size  = batch_size
        self.current_ind = 0

        if pad_with_last_sample :
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding   = np.repeat(xs[-1:], num_padding, axis=0)  # Repeat the last sample
            y_padding   = np.repeat(ys[-1:], num_padding, axis=0)  # y也复制这么多？
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.xs = xs
        self.ys = ys
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)


    def shuffle(self) :

        permutation = np.random.permutation(self.size)  # 类似句柄？
        xs, ys  = self.xs[permutation], self.ys[permutation]
        self.xs = xs  # 原始数据就被打乱，而不是以乱取
        self.ys = ys


    def get_iterator(self) :

        self.current_ind = 0  # 批号

        def _wrapper() :

            while self.current_ind < self.num_batch :

                start_ind = self.batch_size*self.current_ind  # ？
                end_ind = min(self.size, self.batch_size*(self.current_ind+1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


# 
class StandardScaler :


    def __init__(self, mean, std) :
        self.mean = mean  # 均值
        self.std = std    # 标准差

    def transform(self, data) :
        return (data - self.mean) / self.std

    def inverse_transform(self, data) :
        return (data * self.std) + self.mean


# 
def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None) :

    data = {}

    for category in ["train", "val", "test"] :
        cat_data = np.load(os.path.join(dataset_dir, category+".npz"))  # .npz是NumPy库用来存储多个NumPy数组
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]

    scaler = StandardScaler(
        mean = data["x_train"][..., 0].mean(), 
        std  = data["x_train"][..., 0].std()
    )

    # Data format
    for category in ["train", "val", "test"] :
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])  # 三个集以训练集为基准

    # 对顺序出现的数据全局随机打乱
    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))  # 生成一个从0到data["x_train"].shape[0]的序列
    random_train = torch.randperm(random_train.size(0))  # .shape[0]与.size(0)等价，第零维的大小
    data["x_train"] = data["x_train"][random_train, ...]  # 只有前面random_train个被打乱？
    data["y_train"] = data["y_train"][random_train, ...]

    random_val = torch.arange(int(data["x_val"].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]

    # random_test = torch.arange(int(data['x_test'].shape[0]))
    # random_test = torch.randperm(random_test.size(0))
    # data['x_test'] = data['x_test'][random_test, ...]
    # data['y_test'] = data['y_test'][random_test, ...]

    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"]   = DataLoader(data["x_val"]  , data["y_val"]  , valid_batch_size)
    data["test_loader"]  = DataLoader(data["x_test"] , data["y_test"] , test_batch_size)
    data["scaler"] = scaler

    return data


# 
def MAE_torch(pred, true, mask_value=None) :

    if mask_value != None :
        mask = torch.gt(true, mask_value)  # ge/gt/le/lt/ne/eq分别是>=/>/<=/</==/!=
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)

    return torch.mean(torch.abs(true - pred))


# 
def MAPE_torch(pred, true, mask_value=None) :

    if mask_value != None :
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)

    return torch.mean(torch.abs(torch.div((true - pred), true)))


# 
def RMSE_torch(pred, true, mask_value=None) :

    if mask_value != None :
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)

    return torch.sqrt(torch.mean((pred - true) ** 2))


# 
def WMAPE_torch(pred, true, mask_value=None) :

    if mask_value != None :
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))

    return loss


# 
def metric(pred, real) :

    mae   = MAE_torch  (pred, real, 0.0).item()
    mape  = MAPE_torch (pred, real, 0.0).item()
    rmse  = RMSE_torch (pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()

    return mae, mape, rmse, wmape

