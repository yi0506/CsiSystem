"""获取、处理CSI系统仿真数据"""
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import h5py
import numpy as np

from libs import config


class CsiDataset(Dataset):
    """CSI反馈的信道数据集"""
    
    data = None  # 数据
    
    class Configuration:
        drop_last = False  # 丢弃最后一个不足batch size的数据集
        shuffle = True  # 是否打乱数据集
        collate_fn = None  # 对一个data_loader中batch size个数据的进行操作的函数指针
        batch_size = 1  # 一个读取的数据量

    def get_data_loader(self, **settings):
        """获取data_loader"""
        return DataLoader(dataset=self, batch_size=self.Configuration.batch_size, collate_fn=self.Configuration.collate_fn,
                          drop_last=self.Configuration.drop_last, shuffle=self.Configuration.shuffle, **settings)


class COMM_Dataset(CsiDataset):
    """一般环境数据集""" 
    def __getitem__(self, idx):
        # 输入数据实shape为 [2048, ]
        return self.data[idx]

    def __len__(self):
        """数据集大小"""
        return self.data.shape[0]


class HS_Dataset(CsiDataset):
    """高速环境神经网络数据集"""
        
    def __getitem__(self, idx):
        real, img = self.data
        # 输入数据实shape为 [2, 32, 32]
        return np.concatenate((np.expand_dims(real[idx], axis=0), np.expand_dims(img[idx], axis=0)), axis=0)

    def __len__(self):
        return self.data[0].shape[0]
    
    def get_data(self, file_path):
        return [h5py.File(file_path, "r")["save_H_real"][:], h5py.File(file_path, "r")["save_H_img"][:]]


class COMM_CSDataset(COMM_Dataset):
    """压缩感知数据集"""

    def __init__(self):
        FILE_PATH = r"{}/data/cost2100/DATA_Htestin.npy".format(config.BASE_DIR)
        self.data = np.load(FILE_PATH)
        self.Configuration.collate_fn = self.collate_fn

    def collate_fn(self, batch):
        # 返回 [2048, 1]
        return batch[0].reshape(-1, 1)


class COMM_CSINetDataset(COMM_Dataset):
    """一般环境CSINet数据集"""
    def __init__(self, is_train) -> None:
        """
        :param is_train: 是否取训练集
        """
        FILE_PATH = r"{}/data/cost2100/DATA_Htrainin.npy".format(config.BASE_DIR) if is_train else r"{}/data/cost2100/DATA_Htestin.npy".format(config.BASE_DIR)
        self.Configuration.batch_size = config.train_batch_size if is_train is True else config.test_batch_size
        self.data = np.load(FILE_PATH)
        self.Configuration.collate_fn = self.collate_fn

    def collate_fn(self, batch):
        # 返回[batch, 2048]
        return torch.FloatTensor(batch)
    

class COMM_CSINetStuDataset(COMM_CSINetDataset):
    """与COMM_CSINetDataset数据集相同"""


class CSPNetDataset(COMM_Dataset):
    """CSPNet 数据集"""

    def __init__(self, is_train, ratio) -> None:
        """
        :param is_train: 是否取训练集
        """
        FILE_PATH = r"{}/data/cost2100/DATA_Htrainin.npy".format(config.BASE_DIR) if is_train else r"{}/data/cost2100/DATA_Htestin.npy".format(config.BASE_DIR)
        self.ratio = ratio
        self.y = None
        self.y_flag = False
        self.Configuration.collate_fn = self.collate_fn
        self.data = np.load(FILE_PATH)

    def collate_fn(self, batch):
        target = torch.FloatTensor(batch).view(self.Configuration.batch_size, -1)
        if not self.y_flag:
            # 获得观测向量
            Fi_m = target.shape[-1] // self.ratio
            Fi = torch.randn(target.shape[-1], Fi_m, dtype=torch.float32)
            self.y = torch.mm(target, Fi)
            self.y_flag = True
        # 返回 [batch, 2048], [batch, 2048/ratio]
        return target, self.y


class HS_CSDataset(HS_Dataset):
    """高速移动环境下，压缩感知数据集"""

    def __init__(self, velocity):
        FILE_PATH = r"{}/data/matlab/test_100_32_{}_H.mat".format(config.BASE_DIR, velocity)
        self.data = self.get_data(FILE_PATH)
        self.Configuration.collate_fn = self.collate_fn

    def collate_fn(self, batch):
        return batch[0].reshape(-1, 1)


class RMNetDataset(HS_Dataset):
    """RMNet 数据集"""

    def __init__(self, is_train, velocity) -> None:
        """
        :param is_train: 是否取训练集
        :param velocity: 速度
        """
        dataset = "test_10000_32_{}_H.mat".format(velocity) if is_train else r"test_100_32_{}_H.mat".format(velocity)
        FILE_PATH = r"{}/data/matlab/{}".format(config.BASE_DIR, dataset)
        self.data = self.get_data(FILE_PATH)
        self.Configuration.collate_fn = self.collate_fn
        self.Configuration.batch_size = config.train_batch_size if is_train is True else config.test_batch_size

    def collate_fn(self, batch):
        # 返回[batch, 2048]
        return torch.FloatTensor(batch).view(self.Configuration.batch_size, -1)


class RMStuNetDataset(RMNetDataset):
    """RMStuNet 数据集，两者数据集相同"""
    pass


class HS_CSINetDataset(HS_Dataset):
    """CSINet 数据集"""

    def __init__(self, is_train, velocity) -> None:
        """
        :param is_train: 是否取训练集
        :param velocity: 速度
        """
        dataset = "test_10000_32_{}_H.mat".format(velocity) if is_train else r"test_100_32_{}_H.mat".format(velocity)
        FILE_PATH = r"{}/data/matlab/{}".format(config.BASE_DIR, dataset)
        self.data = self.get_data(FILE_PATH)
        self.Configuration.batch_size = config.train_batch_size if is_train is True else config.test_batch_size
        self.Configuration.collate_fn = self.collate_fn

    def collate_fn(self, batch):
        # 返回[batch, 2, 32, 32]
        return torch.FloatTensor(batch).view(self.Configuration.batch_size, -1)


class HS_CSINetStuDataset(HS_CSINetDataset):
    """CsiStuNet 数据集，两者数据集相同"""
    pass


if __name__ == '__main__':
    data_loader = HS_CSINetDataset(True, 50).get_data_loader()
    print(len(data_loader))
    for idx, x in enumerate(tqdm(data_loader)):
        print(x)
        print(x.size())
        break
