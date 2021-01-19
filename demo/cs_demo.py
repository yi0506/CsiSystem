# -*- coding: UTF-8 -*-
"""
@author:wsy123
@file:cs_demo.py
@time:2020/07/30
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from demo.OMP_demo import omp_reform,omp_process
import h5py


def initial_data():
    sparse_A = pickle.load(open("../data/dct_32.pkl", "rb"))  # 稀疏基 [32*32, 32*32]
    dim = sparse_A.shape[0]
    ratio = 512  # 压缩比例
    x = np.arange(dim)
    k = 100

    data = h5py.File("../data/dataset_1000.h5", "r")["H"][2]
    index_k = h5py.File("../data/dataset_1000.h5", "r")["index_k"][2]
    plt.figure(dpi=100)
    plt.stem(x, data, linefmt="r-", use_line_collection=True)
    plt.title("原始数据：data")
    plt.show()

    s = np.matmul(sparse_A, data)
    plt.figure(dpi=100)
    plt.stem(x, s, linefmt="r-", use_line_collection=True)
    plt.title("稀疏系数：s")
    plt.show()

    # 去掉数值小的数据，方便与恢复出的稀疏系数对比
    temp = np.sort(np.abs(s), axis=0)[dim - k]
    temp_data = s
    temp_data[np.abs(s) < temp] = 0
    plt.figure(dpi=100)
    plt.stem(x, temp_data, linefmt="r-", use_line_collection=True)
    plt.title("稀疏系数去掉较小的数：s")
    plt.show()

    Fi = np.random.normal(0, 1, (ratio, dim))  # 设置观测矩阵
    Beta = np.matmul(Fi, sparse_A)  # 传感矩阵
    theta = np.matmul(Fi, data)  # 观测向量
    return Beta, theta, sparse_A, k, index_k


def cs_demo():
    """cs原理流程"""
    Beta, theta, sparse_A, k, index_k = initial_data()
    data_restore, s_restore = omp_reform(theta=theta, Beta=Beta, A=sparse_A, index_k=index_k)
    data_restore, s_restore = omp_process(y=theta, Beta=Beta, A=sparse_A, k=k, index_k=index_k)

    x = np.arange(data_restore.shape[0])

    plt.figure(dpi=100)
    plt.stem(x, s_restore, linefmt="k-", use_line_collection=True)
    plt.title("恢复的s")
    plt.show()
    #
    plt.figure(dpi=100)
    plt.stem(x, data_restore, linefmt="r-", use_line_collection=True)
    plt.title("恢复的data")
    plt.show()


if __name__ == '__main__':
    cs_demo()
