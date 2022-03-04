# -*- coding: UTF-8 -*-
"""各类压缩感知算法的封装并集成在CSI反馈系统中"""
import numpy as np
import torch
import pickle
import tqdm
import os
from abc import ABCMeta, abstractmethod

from . import config
from .cs_restore_method import OMP, SP, SAMP, generate_fft_sparse_base, generate_dct_sparse_base, generate_eye_sparse_base
from .utils import nmse, load_Phi, cal_capacity


class CSConfiguration(object):
    cs_data_length = 2048
    samp_t = 1  # samp算法步长
    # 稀疏基
    sparse_dct = "dct"
    sparse_fft = "fft"
    sparse_eye = "eye"
    # CS恢复算法
    omp = "omp"
    samp = "samp"
    sp = "sp"


class BaseCS(metaclass=ABCMeta):
    """
    CS基类

    对数据进行压缩感知重构，并评估重构结果，每次只能对一条数据进行测试，
    同时，每次测试只能在一种信噪比下使用一种CS的组合方法，进行测试


    算法：
        DCT: 离散余弦变换
        IDCT: 离散余弦逆变换
        FFT：快速傅里叶变换
        OMP：正交匹配追踪
        SAMP：自适应匹配追踪算法
        SP：子空间追踪
    """

    def __init__(self, **kwargs):
        """
        参数与变量:
                y: 观测向量 [m, 1]
                data: 原始信号向量 [n, 1]
                s: data在稀疏基A下的稀疏表示
                k: 稀疏度，非零个数
                Beta: 传感矩阵
                snr: 信噪比
                sparse: 压缩方法
                restore: 重构方法
                A: 稀疏基矩阵
                Fi: 观测矩阵
                t: 步长


        初始化属性:
                self.sparse  # 确定稀疏基
                self.restore   # 确定重构算法
                self.snr  # 信噪比
                self.ratio  # 压缩率
                self.data_loader = data_loader  # 数据集迭代器

        """
        # 初始化实例属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.sparse_matrix = self.load_sparse_base(self.sparse.upper())  # 加载稀疏基

    @staticmethod
    def load_sparse_base(method):
        if method == "DCT":
            if not os.path.exists("./data/CS/dct_sp.pkl"):
                generate_dct_sparse_base(CSConfiguration.cs_data_length)
            return pickle.load(open("./data/CS/dct_sp.pkl", "rb"))  # 获得稀疏基
        elif method == "FFT":
            if not os.path.exists("./data/CS/fft_sp.pkl"):
                generate_fft_sparse_base(CSConfiguration.cs_data_length)
            return pickle.load(open("./data/CS/fft_sp.pkl", "rb"))  # 获得稀疏基
        elif method == "EYE":
            return generate_eye_sparse_base(CSConfiguration.cs_data_length)
        else:
            exit("压缩方法错误")

    def run(self):
        """主函数：对数据进行压缩、重构、计算相似度与损失，然后返回测试结果"""
        similarity_list = list()
        nmse_list = list()
        capacity_list = list()
        bar = tqdm.tqdm(self.data_loader)
        for data in bar:
            # 数据压缩
            y, Beta, k = self.__compress(data)
            # 经过信道加噪
            y_add_noise = self.__channel(y)
            # 重构信号
            restore_data = self.__restore(y_add_noise, Beta, k)
            # 计算相似度、损失、消耗的时间
            similarity, nmse, capacity = self.__evaluate(data, restore_data)
            similarity_list.append(similarity)
            nmse_list.append(nmse)
            capacity_list.append(capacity)
            # 显示进度
            bar.set_description("ratio:{}\t{}:{}\tnoise_SNR:{}dB\tnmse:{}\tsimilarity:{:.3f}\tcapacity:{}".format(self.ratio, self.sparse, self.restore, self.snr, nmse, similarity, capacity))
        # 将测试结果返回
        return self.__cal_eval_result(nmse_list, similarity_list, capacity_list)

    def __compress(self, data):
        """
        压缩数据:
            获得稀疏基与观测矩阵Fi, 通过计算, 返回传感矩阵与观测向量
        :param data:[?, 1]
        :return: 传感矩阵beta, 压缩后的数据y,稀疏度k
        """
        Phi_m = CSConfiguration.cs_data_length // self.ratio  # 得到观测矩阵的行数
        dim = data.shape[0]
        # 加载观测矩阵
        Phi_path = "{}/data/CS/Phi_{}.npy".format(config.BASE_DIR, Phi_m)
        Phi = load_Phi(Phi_path)
        y = np.matmul(Phi, data)  # 观测向量
        Beta = np.matmul(Phi, self.sparse_matrix)  # 传感矩阵
        # 确定稀疏度k
        k = np.sum(data >= 1e-3)
        return y, Beta, k

    def __channel(self, y):
        """
        信号加入高斯白噪声,
        噪音强度为：10 ** (snr / 10)，其中snr为对应的信噪比
        注：信号的幅度的平方==信号的功率，因此要开平方根
        :return: y + 噪声
        """
        # 产生对应信噪比的高斯白噪声
        if y.dtype == np.complex:
            return self.__gs_noise(np.real(y), self.snr) + 1j * self.__gs_noise(np.imag(y), self.snr)
        else:
            return self.__gs_noise(y, self.snr)

    def __restore(self, y_add_noise, Beta, k):
        """
        对y进行重构，恢复成原始数据
        :param y_add_noise: 经过信道后的压缩数据
        :param Beta: 传感矩阵
        :return: restore_data:重构后的数据
        func不指向route中的call_func，指向被修饰的不同实例方法
        """
        if self.sparse == "dct":
            return self.DCT_restore(y_add_noise, Beta, k)
        elif self.sparse == "fft":
            return self.FFT_restore(y_add_noise, Beta, k)
        elif self.sparse == "eye":
            return self.EYE_restore(y_add_noise, Beta, k)
        else:
            raise ValueError("只能写dft, fft, eye")

    def __evaluate(self, data, refine_data):
        """评估余弦相似度、mse损失"""
        data = torch.from_numpy(data)
        refine_data = torch.from_numpy(refine_data)
        nmse_ = nmse(refine_data, data, "torch")
        similarity = torch.cosine_similarity(data, refine_data, dim=0).item()
        capacity = cal_capacity(refine_data, self.snr)
        return similarity, nmse_, capacity

    def __cal_eval_result(self, nmse_list, similarity_list, capacity_list):
        """计算平局损失、平均相似度与平均计算时间"""
        avg_nmse = np.mean(nmse_list)
        avg_similarity = np.mean(similarity_list)
        avg_capacity = np.mean(capacity_list)

        # 保存结果到字典中并返回
        return {"snr": self.snr, "NMSE": avg_nmse, "相似度": avg_similarity, "Capacity": avg_capacity}

    @staticmethod
    def __gs_noise(y, snr):
        """加入高斯白噪声"""
        if snr is None:
            return y
        y_power = (np.sum(y ** 2) / y.size)  # 信号的功率
        noise_power = (y_power / (10 ** (snr / 10)))  # 噪声的功率
        gaussian = np.random.normal(0, np.sqrt(noise_power), y.shape)  # 产生对应信噪比的高斯白噪声
        return gaussian + y

    def FFT_restore(self, y_add_noise, Beta, k):
        """
        基于FFT稀疏基的CS重构算法
        func: 需要执行的具体CS算法
        param: 执行func所指向的算法所需的必要参数
        """
        Beta_real = np.real(Beta)
        Beta_imag = np.imag(Beta)
        # 进行CS恢复计算
        restore_s_real = self.CS_p(y_add_noise, Beta_real, k)  # s实部
        restore_s_imag = self.CS_p(y_add_noise, Beta_imag, k)  # s虚部
        # 实虚部合并
        restore_s = restore_s_real + restore_s_imag * 1j
        # IFFT变换，恢复data
        restore_data = np.matmul(self.sparse_matrix.T, restore_s)
        # 取实部，作为恢复的信号
        return np.real(restore_data)

    def DCT_restore(self, y_add_noise, Beta, k):
        """
        基于DCT稀疏基的CS重构算法
        func: 需要执行的具体CS算法
        param: 执行func所指向的算法所需的必要参数
        """
        restore_s = self.CS_p(y_add_noise, Beta, k)
        restore_data = np.matmul(self.sparse_matrix.T, restore_s)
        return restore_data

    def EYE_restore(self, y_add_noise, Beta, k):
        """
        不需要稀疏变换的CS重构算法
        func: 需要执行的具体CS算法
        param: 执行func所指向的算法所需的必要参数
        """
        return self.CS_p(y_add_noise, Beta, k)

    @abstractmethod
    def CS_p(self, y_add_noise, Beta, k):
        """需要实现的CS算法"""
        pass

    def __call__(self, *args, **kwargs):
        return self.run()


class OMPCS(BaseCS):
    """基于OMP的CS方法"""

    def CS_p(self, y_add_noise, Beta, k, *args):
        return OMP(y_add_noise, Beta, k)


class SPCS(BaseCS):
    """基于SP的CS方法"""

    def CS_p(self, y_add_noise, Beta, k, *args):
        return SP(y_add_noise, Beta, k)


class SAMPCS(BaseCS):
    """基于SAMP的CS算法"""

    def CS_p(self, y_add_noise, Beta, k, *args):
        # 返回恢复后的数据
        return SAMP(y_add_noise, Beta, CSConfiguration.samp_t)


if __name__ == '__main__':
    pass
