# -*- coding: UTF-8 -*-
"""
@author:wsy123
@file:OMP_demo.py
@time:2020/07/16
"""
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.inf)  # 设置输出数组元素的数目
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def omp_process(y, Beta, A, k):
    """
    未知索引，重构信号
    :param y: 观测向量 [m, 1]
    :param Beta: 传感矩阵 [m, n], m << n
    :param k: 稀疏度
    :param A: 稀疏基
    :return: 重构数据 restore_data[n, 1]
    """
    Beta_row, Beta_col = Beta.shape

    iter_num = k  # 确定迭代次数
    s = np.zeros((Beta_col, 1))  # 数据的稀疏系数
    Beta_new = np.zeros((Beta_row, iter_num))  # 用来存储迭代过程中Beta被选择的列
    Beta_idx = np.zeros((iter_num, 1)).astype(np.int32)  # 存储Beta被选择的列的索引
    residual_error = y  # 初始化残差

    for i in range(iter_num):
        # 传感矩阵与各列残差求内积
        inner_product = np.matmul(Beta.T, residual_error)

        # 找到最大的内积绝对值相对应列的索引
        idx = np.argmax(np.abs(inner_product))

        # 存储这一列与对应的索引
        Beta_new[:, i] = Beta[:, idx]
        Beta_idx[i] = idx

        # 清零Beta的这一列， 其实可以不用，因为它与残差正交
        Beta[:, idx] = np.zeros((Beta_row, ))

        # 求s的最小二乘解,所求结果为s中不为0的位置处的值
        s_ls = np.matmul(np.linalg.pinv(Beta_new[:, :i+1]), y)

        # 更新残差
        residual_error = y - np.matmul(Beta_new[:, :i + 1], s_ls)

    # 得到s的稀疏系数
    s[Beta_idx.flatten()] = s_ls

    # 重构信号
    restore_data = np.matmul(A.T, s)
    return restore_data, s


def omp_process_know_index(theta, Beta, A, index_k):
    """已知索引，重构信号"""
    Beta_row, Beta_col = Beta.shape

    iter_num = index_k.shape[0]  # 确定迭代次数
    s = np.zeros((Beta_col, 1))  # 数据的稀疏系数
    Beta_new = np.zeros((Beta_row, iter_num))  # 用来存储迭代过程中Beta被选择的列
    Beta_idx = np.zeros((iter_num, 1)).astype(np.int32)  # 存储Beta被选择的列的索引

    for i in range(iter_num):

        # 让算法知道索引在哪个位置
        idx = index_k[i].item()

        # 存储这一列与对应的索引
        Beta_new[:, i] = Beta[:, idx]
        Beta_idx[i] = idx

    # 求s的最小二乘解,所求结果为s中不为0的位置处的值
    s_ls = np.matmul(np.linalg.pinv(Beta_new), theta)

    # 得到data的稀疏系数
    s[Beta_idx.flatten()] = s_ls
    restore_data = np.matmul(A.T, s)
    return restore_data, s


def initial_data():
    m = 100  # 原始信号长度
    n = 50  # 观测向量长度
    k = 20  # 稀疏度
    data = np.zeros((m, 1))  # 假设信号本身就是完美稀疏的
    data = np.random.normal(0, 0.2, (m, 1))  # 假设信号本身就是稀疏的
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

    data_restore, s_restore = omp_process(y=theta, Beta=Beta, A=A, k=k, index_k=index_k)

    plt.figure(dpi=100)
    plt.stem(np.arange(data_restore.shape[0]), data_restore, linefmt="r-", use_line_collection=True)
    plt.title("恢复信号")
    plt.show()


