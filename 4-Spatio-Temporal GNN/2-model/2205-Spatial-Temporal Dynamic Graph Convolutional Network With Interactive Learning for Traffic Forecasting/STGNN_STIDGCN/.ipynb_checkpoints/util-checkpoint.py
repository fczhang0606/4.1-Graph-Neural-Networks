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
            # print(num_padding)  # 53 +10699=64*168
            x_padding   = np.repeat(xs[-1:], num_padding, axis=0)  # Repeat the last sample
            # print(x_padding)  # (53, 12, 170, 3)
            y_padding   = np.repeat(ys[-1:], num_padding, axis=0)  # y也复制这么多？
            # print(y_padding)  # (53, 12, 170, 1)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.xs = xs
        self.ys = ys
        self.size = len(xs)  # 10752
        self.num_batch = int(self.size // self.batch_size)  # 168


    def shuffle(self) :

        permutation = np.random.permutation(self.size)  # 类似句柄？
        xs, ys  = self.xs[permutation], self.ys[permutation]
        self.xs = xs  # 原始数据就被打乱，而不是以乱取
        self.ys = ys


    def get_iterator(self) :  # 调用一次，返回168批

        self.current_ind = 0  # 批号

        def _wrapper() :  # 内嵌的装饰器函数，拓展功能

            while self.current_ind < self.num_batch :

                start_ind = self.batch_size*self.current_ind
                end_ind = min(self.size, self.batch_size*(self.current_ind+1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)  # 类似return，节约内存

                self.current_ind += 1

        return _wrapper()  # 返回函数


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
def load_dataset(dataset_dir, train_batch_size, valid_batch_size=None, test_batch_size=None) :

    data = {}

    # 6=3*2
    for category in ["train", "val", "test"] :
        cat_data = np.load(os.path.join(dataset_dir, category+".npz"))  # .npz是NumPy库用来存储多个NumPy数组
        data["x_" + category] = cat_data["x"]
        # print(data["x_" + category].shape)  # (10699/3567/3567, 12, 170, 3)
        data["y_" + category] = cat_data["y"]
        # print(data["y_" + category].shape)  # (10699/3567/3567, 12, 170, 1)

    # class
    scaler = StandardScaler(
        # https://blog.csdn.net/g944468183/article/details/124473886
        mean = data["x_train"][..., 0].mean(),  # 均以训练集为基准
        std  = data["x_train"][..., 0].std()
    )
    # print(data["x_train"][..., 0].shape)  # (10699, 12, 170)
    # print(data["x_train"][..., 0].mean())  # 229.85893440655073
    # print(data["x_train"][..., 0].std())   # 145.62268077938813

    # Data format
    for category in ["train", "val", "test"] :
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])  # 对三个集合的x部分进行缩放

    # 对顺序出现的数据全局随机打乱
    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))  # 生成一个从0到data["x_train"].shape[0]的序列
    # print(int(data["x_train"].shape[0]))  # 10699
    random_train = torch.randperm(random_train.size(0))  # .shape[0]与.size(0)等价，第零维的大小
    # print(random_train)
    data["x_train"] = data["x_train"][random_train, ...]  # 四维的第一维重排
    data["y_train"] = data["y_train"][random_train, ...]

    random_val = torch.arange(int(data["x_val"].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]

    # random_test = torch.arange(int(data['x_test'].shape[0]))
    # random_test = torch.randperm(random_test.size(0))
    # data['x_test'] = data['x_test'][random_test, ...]
    # data['y_test'] = data['y_test'][random_test, ...]

    # data[]是自己做出来的
    # print(data)
    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], train_batch_size)
    data["val_loader"]   = DataLoader(data["x_val"]  , data["y_val"]  , valid_batch_size)
    data["test_loader"]  = DataLoader(data["x_test"] , data["y_test"] , test_batch_size)
    data["scaler"] = scaler

    # print(data)  # 浪费内存？

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

