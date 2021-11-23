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

    def get_data(self):
        return h5py.File(self.FILE_PATH, "r")["save_H_real"], h5py.File(self.FILE_PATH, "r")["save_H_img"]

    def get_data_loader(self, **settings):
        """获取data_loader"""
        return DataLoader(dataset=self, 
                          batch_size=self.Configuration.batch_size, 
                          num_workers=self.Configuration.num_workers, 
                          collate_fn=self.Configuration.collate_fn, 
                          drop_last=self.Configuration.drop_last, 
                          shuffle=self.Configuration.shuffle, 
                          **settings)

    def __getitem__(self, idx):
        real, img = self.get_data()
        # 输入数据实shape为 [2, 32, 32]
        return np.concatenate((np.expand_dims(real[idx], axis=0), np.expand_dims(img[idx], axis=0)), axis=0)

    def __len__(self):
        return self.get_data()[0].shape[0]


class CSDataset(CsiDataset):
    """压缩感知数据集"""

    def __init__(self, velocity):
        self.FILE_PATH = r"{}/data/matlab/test_100_32_{}_H.mat".format(config.BASE_DIR, velocity)


class NetDataset(CsiDataset):
    """神经网络数据集"""

    def __init__(self, is_train) -> None:
        self.Configuration.batch_size = config.train_batch_size if is_train is True else config.test_batch_size


class RMNetDataset(NetDataset):
    """RM Net 数据集"""

    def __init__(self, is_train, velocity) -> None:
        """
        :param is_train: 是否取训练集
        :param velocity: 速度
        """
        super().__init__(is_train)
        dataset = "test_10000_32_{}_H.mat".format(velocity) if is_train else r"test_100_32_{}_H.mat".format(velocity)
        self.FILE_PATH = r"{}/data/matlab/{}".format(config.BASE_DIR, dataset)
        self.Configuration.collate_fn = self.collate_fn

    def collate_fn(self, batch):
        return torch.FloatTensor(batch).view(self.Configuration.batch_size, -1)


class CSINetDataset(NetDataset):
    """CSI Net 数据集"""

    def __init__(self, is_train, velocity) -> None:
        """
        :param is_train: 是否取训练集
        :param velocity: 速度
        """
        super().__init__(is_train)
        dataset = "test_10000_32_{}_H.mat".format(velocity) if is_train else r"test_100_32_{}_H.mat".format(velocity)
        self.FILE_PATH = r"{}/data/matlab/{}".format(config.BASE_DIR, dataset)
        self.Configuration.collate_fn = self.collate_fn

    def collate_fn(self, batch):
        return torch.FloatTensor(batch)


if __name__ == '__main__':
    data_loader = RMNetDataset(True, 50).get_data_loader()
    print(len(data_loader))
    for i in tqdm(data_loader):
        print(i.size())
