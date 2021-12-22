"""获取、处理CSI系统仿真数据"""
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import h5py
import numpy as np

from libs import config


class CsiDataset(Dataset):
    """CSI反馈的信道数据集"""

    FILE_PATH = None  # 数据集的路径

    class Configuration:
        num_workers = 0  # 读取数据的线程数
        drop_last = False  # 丢弃最后一个不足batch size的数据集
        shuffle = True  # 是否打乱数据集
        collate_fn = None  # 对一个data_loader中batch size个数据的进行操作的函数指针
        batch_size = 1  # 一个读取的数据量

    def get_data_loader(self, **settings):
        """获取data_loader"""
        return DataLoader(dataset=self, batch_size=self.Configuration.batch_size,
                          num_workers=self.Configuration.num_workers, collate_fn=self.Configuration.collate_fn,
                          drop_last=self.Configuration.drop_last, shuffle=self.Configuration.shuffle, **settings)
    
    def get_data(self):
        """获取数据"""
        pass


class COMM_Dataset(CsiDataset):
    """一般环境数据集"""
    
    def get_data(self):
        return np.load(self.FILE_PATH)
    
    def __getitem__(self, idx):
        # 输入数据实shape为 [2048, ]
        return self.get_data()[idx]

    def __len__(self):
        """数据集大小"""
        return self.get_data().shape[0]


class COMM_CSDataset(COMM_Dataset):
    """压缩感知数据集"""

    def __init__(self):
        self.FILE_PATH = r"{}/data/cost2100/DATA_Htestin.npy".format(config.BASE_DIR)
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
        self.FILE_PATH = r"{}/data/cost2100/DATA_Htrainin.npy".format(config.BASE_DIR) if is_train else r"{}/data/cost2100/DATA_Htestin.npy".format(config.BASE_DIR)


class COMM_CSINetStuDataset(COMM_CSINetDataset):
    """与COMM_CSINetDataset数据集相同"""


class CSPNetDataset(COMM_Dataset):
    """CSPNet 数据集"""

    def __init__(self, is_train, ratio) -> None:
        """
        :param is_train: 是否取训练集
        """
        self.FILE_PATH = r"{}/data/cost2100/DATA_Htrainin.npy".format(config.BASE_DIR) if is_train else r"{}/data/cost2100/DATA_Htestin.npy".format(config.BASE_DIR)
        self.ratio = ratio
        self.y = None
        self.y_flag = False
        self.Configuration.collate_fn = self.collate_fn

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


class HS_Dataset(CsiDataset):
    """高速环境神经网络数据集"""
        
    def get_data(self):
        return h5py.File(self.FILE_PATH, "r")["save_H_real"], h5py.File(self.FILE_PATH, "r")["save_H_img"]
    
    def __getitem__(self, idx):
        real, img = self.get_data()
        # 输入数据实shape为 [2, 32, 32]
        return np.concatenate((np.expand_dims(real[idx], axis=0), np.expand_dims(img[idx], axis=0)), axis=0)

    def __len__(self):
        return self.get_data()[0].shape[0]


class HS_CSDataset(HS_Dataset):
    """高速移动环境下，压缩感知数据集"""

    def __init__(self, velocity):
        self.FILE_PATH = r"{}/data/matlab/test_100_32_{}_H.mat".format(config.BASE_DIR, velocity)
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
        self.FILE_PATH = r"{}/data/matlab/{}".format(config.BASE_DIR, dataset)
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
        self.FILE_PATH = r"{}/data/matlab/{}".format(config.BASE_DIR, dataset)
        self.Configuration.batch_size = config.train_batch_size if is_train is True else config.test_batch_size


class HS_CSINetStuDataset(HS_CSINetDataset):
    """CsiStuNet 数据集，两者数据集相同"""
    pass


if __name__ == '__main__':
    data_loader = CSPNetDataset(True, 8).get_data_loader()
    print(len(data_loader))
    for idx, (target, y) in enumerate(tqdm(data_loader)):
        print(target, y)
        print(target.size(), y.size())
        break
