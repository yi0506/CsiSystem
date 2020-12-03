"""获取、处理CSI系统仿真数据"""
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import h5py
import numpy as np
import pickle

from libs import config


class CsiDataset(Dataset):
    """获取CSI反馈的信道数据集"""
    def __init__(self, is_train, velocity):
        self.is_train = is_train  # 是否取训练集
        current_path = ".." if __name__ == "__main__" else "."
        file_path = r"{}/data/matlab/test_10000_32_{}_H.mat".format(current_path, velocity) if self.is_train else r"{}/data/matlab/test_1000_32_{}_H.mat".format(current_path, velocity)
        self.data = h5py.File(file_path, "r")["save_H"]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    @property
    def size(self):
        """数据集的尺寸"""
        return self.data.shape


def data_load(is_train, velocity):
    """数据集生成器"""
    batch_size = config.train_batch_size if is_train is True else config.test_batch_size
    return DataLoader(dataset=CsiDataset(is_train, velocity), batch_size=batch_size, num_workers=0, drop_last=True,
                      collate_fn=lambda x: torch.FloatTensor(x).view(batch_size, -1), shuffle=config.shuffle)


def data_generator(k=100, num=1000):
    """生成在角度域稀疏系数为k的data"""
    sparse_A = pickle.load(open("../data/dct_32.pkl", "rb"))  # 稀疏基 [32*32, 32*32]
    dim = sparse_A.shape[0]
    data_list = list()
    index_list = list()
    for i in range(num):
        # 生成一个符合稀疏基稀疏之后的稀疏系数
        s_coefficient = np.random.normal(0, 0.2, (dim, 1))
        index_k = np.random.choice(k * 3, (k, 1), replace=False)
        _temp = np.random.normal(0, 2, (k, 1))
        s_coefficient[index_k.flatten()] = _temp

        # 通过稀疏系数，反求出data,保存生成的data
        data = np.matmul(sparse_A.T, s_coefficient)
        data_list.append(data)
        index_list.append(index_k)
    dataset = np.array(data_list)

    # 保存数据
    f = h5py.File("./data/dataset_{}.h5".format(num), "w")
    f.create_dataset("H", data=dataset)
    f.create_dataset("index_k", data=index_list)
    f.close()


def save_dct_sparse_base():
    """获得DCT稀疏基"""
    x = np.random.randn(32 * 32, 1)
    dim = x.shape[0]
    A = np.zeros((dim, dim))  # A:稀疏基矩阵

    # 确定稀疏基矩阵系数
    for i in range(dim):
        for j in range(dim):
            if i == 0:
                N = np.sqrt(1 / dim)  # 系数N:保证A为正交矩阵
            else:
                N = np.sqrt(2 / dim)
            A[i][j] = N * np.cos(np.pi * (j + 0.5) * i / dim)
            print("第{}行，第{}列".format(i, j))
    # 保存稀疏基
    pickle.dump(A, open("../data/dct_32.pkl", "wb"))


def save_fft_sparse_base():
    """
    获得FFT稀疏基

    X[k] = $\sum_{n=0}^(N-1)x[n]*exp(-j*2*pi*k*n/N)$
    k: frequency index
    N: length of complex sinusoid in samples
    n: 当前样本
    """
    x = np.random.randn(32 * 32, 1)
    N = x.shape[0]
    n = np.arange(N).reshape(1, N)
    m = n.T * n / N  # [N, 1] * [1, N] = [N, N]
    S = np.exp(-1j * 2 * np.pi * m)

    # 保存稀疏基
    pickle.dump(S, open("../data/fft_32.pkl", "wb"))


if __name__ == '__main__':
    data_loader = data_load()
    print(len(data_loader))
    for i in tqdm(data_loader):
        print(i.size())
