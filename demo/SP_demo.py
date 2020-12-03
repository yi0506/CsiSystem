# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch


np.set_printoptions(threshold=np.inf)  # 设置输出数组元素的数目
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def sp_process(y, Beta, A, k):
    """
    :param y: 观测向量 [m, 1]
    :param Beta: 传感矩阵 [m, n], m << n
    :param k: 稀疏度
    :param A: 稀疏基
    :return: 重构数据 restore_data[n, 1]
    """
    Beta_row, Beta_col = Beta.shape
    iter_num = k  # 确定迭代次数
    s = np.zeros((Beta_col, 1))  # 数据的稀疏系数
    Beta_idx = np.array([]).astype(np.int)  # 存储Beta被选择的列的索引
    res_error = y  # 初始化残差

    for i in range(iter_num):
        inner_product = np.abs(np.matmul(Beta.T, res_error))  # 传感矩阵与各列残差求内积
        inner_product = torch.tensor(inner_product)
        _, idx = torch.topk(inner_product, k=k, dim=0)  # 降序排列，选出最大的k列
        idx = idx.numpy()

        # Is这里的操作是将两个数组取并集，去掉重复数据，并且升序排序，也就是支撑集
        Is = np.union1d(idx, Beta_idx)

        # 防止步长过大，超过矩阵维度,Beta矩阵行数要大于列数，此为最小二乘的基础(列线性无关)
        if len(Is) <= Beta_row:
            Beta_t = Beta[:, Is.flatten()]  # 将Beta的这几列组成矩阵Beta_t
        else:  # Beta_t的列数大于行数，列必为线性相关的,Beta_t' * Beta_t将不可逆
            break

        # y = Beta_t * s，以下求s的最小二乘解(Least Square)
        s_ls = np.matmul(np.linalg.pinv(Beta_t), y)

        # 修建
        s_ls = torch.tensor(s_ls)
        _, idx2 = torch.topk(torch.abs(s_ls), k=k, dim=0)  # 降序排列，选出最大的L个

        # 样本更新
        Beta_idx = Is[idx2.flatten()]
        s_ls = s_ls[idx2.flatten()].numpy()

        # Beta_t[:,idx2.flatten()]*s_ls是y在Beta_t[:,idx2.flatten()]列空间上的正交投影
        res_error = y - np.matmul(Beta_t[:, idx2.flatten()], s_ls)

        # 满足停止的阈值,停止迭代   残差的范数<1e-6
        if np.linalg.norm(res_error, axis=0) < 1e-6:
            break

    # 得到s的稀疏系数
    s[Beta_idx.flatten()] = s_ls

    # 重构信号
    restore_data = np.matmul(A.T, s)
    return restore_data, s


def initial_data():
    m = 100  # 原始信号长度
    n = 50  # 观测向量长度
    k = 10  # 稀疏度
    data = np.random.normal(0, 0.2, (m, 1))  # 假设信号本身就是稀疏的
    data = np.zeros((m, 1))  # 假设信号本身就是完美稀疏的
    index_k = np.random.choice(m, (k, 1), replace=False)  # 设置稀疏度为k
    data[index_k.flatten()] = np.random.normal(0, 2, (k, 1))  # 原始信号，设置为稀疏度为k
    # data[:k] = np.random.normal(0, 2, (k, 1))   # 原始信号，设置为稀疏度为k
    A = np.identity(m)  # 设计稀疏基
    Fi = np.random.normal(0, 1, (n, m))  # 设置观测矩阵
    Beta = np.matmul(Fi, A)  # 传感矩阵
    theta = np.matmul(Fi, data)  # 观测向量


    plt.figure(dpi=100)
    plt.stem(np.arange(m), data, linefmt="r-", use_line_collection=True)
    plt.title("原始信号")
    plt.show()

    return Beta, theta, A, k, index_k


if __name__ == '__main__':
    Beta, theta, A, k, index_k = initial_data()

    data_restore, s_restore = sp_process(y=theta, Beta=Beta, A=A, k=k)

    plt.figure(dpi=100)
    plt.stem(np.arange(data_restore.shape[0]), data_restore, linefmt="r-", use_line_collection=True)
    plt.title("恢复信号")
    plt.show()
