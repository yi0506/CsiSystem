# -*- coding: UTF-8 -*-
"""
@author:wsy123
@file:DCT_demo.py
@time:2020/07/16
"""

import numpy as np
import torch.nn.functional as F
import torch


def dct_1d(x):
    x_row, x_col = x.shape
    dim = max(x.shape)
    A = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == 0:
                beta = np.sqrt(1/dim)  # 系数beta:保证A为正交矩阵
            else:
                beta = np.sqrt(2/dim)
            A[i][j] = beta*np.cos(np.pi*(j+0.5)*i/dim)
    # DCT变换
    A_T = A.T  # A的转置矩阵
    x_transposed = np.matmul(A, x)
    return x_transposed


def dct_process(X, snr=None, ratio=None):
    """
    完成DCT变换与逆变换，同时需要设置指定压缩比例
    :param ratio: 压缩比例
    :param X: 待变换矩阵,x需要为方阵,不是方阵一般需要补0，补齐之后再做变换，反变换之后去掉补齐的部分
    :param snr: 信噪比
    :return:误差loss、相似度similarity
    """
    x_row, x_col = X.shape

    # 矩阵X行列不相等时，进行补齐
    if x_col != x_row:
        dim = max(x_row, x_col)
        temp = np.zeros((dim, dim))
        A = np.zeros((dim, dim))

        if x_col > x_row:
            temp[:x_row, :] = X
            X = temp
        else:
            temp[:, :x_col] = X
            X = temp
    else:
        dim = x_row
        A = np.zeros((dim, dim))  # A:观测矩阵

    # 确定观测矩阵系数
    for i in range(dim):
        for j in range(dim):
            if i == 0:
                beta = np.sqrt(1/dim)  # 系数beta:保证A为正交矩阵
            else:
                beta = np.sqrt(2/dim)
            A[i][j] = beta*np.cos(np.pi*(j+0.5)*i/dim)

    # 数据压缩
    if ratio:
        assert ratio < max(x_col, x_row), "ratio必须小于矩阵X行与列的最大维度，否则压缩无效"
        A = A[:ratio, :]

    # DCT变换
    A_T = A.T  # A的转置矩阵
    X_transposed = np.matmul(np.matmul(A, X), A_T)

    # 经过信道
    if snr:
        nVar = 10 ** (0.1 * (-snr))  # 噪声方差
        X_transposed = X_transposed + np.random.normal(0, nVar, (X_transposed.shape[0], X_transposed.shape[1]))

    # DCT逆变换
    X_new = np.matmul(np.matmul(A_T, X_transposed), A)

    # 去掉补齐的部分
    if x_col != x_row:
        if x_col > x_row:
            X_new = X_new[:x_row, :]
            X = X[:x_row, :]
        else:
            X_new = X_new[:, :x_col]
            X = X[:, :x_col]

    # 计算准确率
    X_new = torch.tensor(X_new).flatten()
    X = torch.tensor(X).flatten()
    loss = F.mse_loss(X, X_new)
    similarity = torch.cosine_similarity(X, X_new, dim=-1)
    return loss.item(), similarity.item(), X, X_new, A, X_transposed


def calculate_correction(A):
    """计算矩阵相关性， A为稀疏基矩阵"""
    fi = np.random.randn(6, 6)
    ret_list = []
    for i in range(6):
        temp = np.matmul(fi[i, :], A[i, :])
        list2 = []
        list3 = []
        for j in range(6):
            temp2 = fi[i, j]*fi[i, j]
            list2.append(temp2)
            temp3 = A[i, j] * A[i, j]
            list3.append(temp3)
        ret = temp/(np.sqrt(sum(list3))*np.sqrt(sum(list2)))
        ret_list.append(ret)
    ret = np.sqrt(6)*max(ret_list)
    print(ret)


if __name__ == '__main__':
    loss, simi, x, x_new, A, x_trans = dct_process(np.random.randn(3, 3))
    # print(loss)
    # print(simi)
    # print(x)
    # print(x_new)
    # print(_)
    # print(A)
    # calculate_correction(A)
    np.random.seed(1)
    x = np.random.randn(10, 1)
    # print(x)
    x_trans = dct_1d(x)
    print(np.sort(x_trans, axis=0))
    k = 4
    print(np.sort(x_trans, axis=0)[10-k])
    temp = np.sort(x_trans, axis=0)[10 - k]
    x_trans[x_trans < temp] = 0
    print(x_trans)
    np.where

