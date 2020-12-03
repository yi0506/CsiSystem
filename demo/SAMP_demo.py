# -*- coding: UTF-8 -*-
"""
@author:wsy123
@file:SAMP_demo.py
@time:2020/07/20
"""
import numpy as np
import matplotlib.pyplot as plt
import torch


np.set_printoptions(threshold=np.inf)  # 设置输出数组元素的数目
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


def samp_process(Beta, y, t, A):
    """
    :param y: 观测向量 [m, 1]
    :param Beta: 传感矩阵 [m, n]
    :param t: 步长
    :param A: 稀疏基
    :return: 重构数据 restore_data[n, 1]
    """
    Beta_row, Beta_col = Beta.shape
    iter_num = Beta_row  # 确定迭代次数
    s = np.zeros((Beta_col, 1))  # 存储恢复的稀疏系数
    Beta_idx = np.array([]).astype(np.int)  # 存储Beta被选择的列的索引
    res_error = y  # 初始化残差
    L = t  # 初始化步长

    for i in range(iter_num):
        inner_product = np.abs(np.matmul(Beta.T, res_error))  # 传感矩阵与各列残差求内积
        inner_product = torch.tensor(inner_product)
        _, idx = torch.topk(inner_product, k=L, dim=0)  # 降序排列，选出最大的L个
        idx = idx.numpy()

        # Ck这里的操作是将两个数组合并，去掉重复数据，并且升序排序，也就是支撑集
        Ck = np.union1d(idx, Beta_idx).astype(np.int)

        # 防止步长过大，超过矩阵维度
        if len(Ck) <= Beta_row:
            Beta_t = Beta[:, Ck.flatten()]
        else:
            s_ls = 0
            break

        # y = Beta_t * s，以下求s的最小二乘解(Least Square)
        s_ls = np.matmul(np.linalg.pinv(Beta_t), y)
        s_ls = torch.tensor(s_ls)
        _, idx2 = torch.topk(s_ls, k=L, dim=0)  # 降序排列，选出最大的L个
        F = Ck[idx2.flatten()]

        # 更新残差
        Beta_Lt = Beta[:, F.flatten()]
        s_ls = np.matmul(np.linalg.pinv(Beta_Lt), y)
        res_new = y - np.matmul(Beta_Lt, s_ls)

        # 满足停止的阈值,停止迭代   残差的范数<1e-6
        if np.linalg.norm(res_error, axis=0) < 1e-6:
            Beta_idx = F
            break  
        # 这里做了一个切换，意思是说，如果新的残差的范数比之前的范数大，说明重构不够精准，差距过大.
        elif np.linalg.norm(res_error) >= np.linalg.norm(res_new):
            L = L + t
            if i+1 == iter_num:  # 最后一次循环
               Beta_idx = F  # 更新Beta_idx与s_ls匹配，防止报错
        else:
            Beta_idx = F
            res_error = res_new
            
    # 得到s的稀疏系数
    s[Beta_idx.flatten()] = s_ls

    # 重构信号
    restore_data = np.matmul(A.T, s)
    return restore_data, s



def initial_data():
    m = 100  # 原始信号长度
    n = 60  # 观测向量长度
    k = 10  # 稀疏度
    # data = np.random.normal(0, 0.2, (m, 1))  # 假设信号本身就是稀疏的
    data = np.zeros((m, 1))  # 假设信号本身就是稀疏的
    index_k = np.random.choice(m, (k, 1), replace=False)  # 设置稀疏度为k
    data[index_k.flatten()] = np.random.normal(0, 1, (k, 1))  # 原始信号，设置为稀疏度为k
    # data[:k] = np.random.normal(0, 2, (k, 1))   # 原始信号，设置为稀疏度为k
    A = np.identity(m)  # 设计稀疏基
    Fi = np.random.normal(0, 1, (n, m))  # 设置观测矩阵
    Beta = np.matmul(Fi, A)  # 传感矩阵
    y = np.matmul(Fi, data)  # 观测向量

    plt.figure(dpi=100)
    plt.stem(np.arange(m), data, linefmt="r-", use_line_collection=True)
    plt.title("原始信号")
    plt.show()
    return Beta, y, A, k, index_k


if __name__ == '__main__':
    m = 100
    Beta, y, A, *_ = initial_data()
    t = 1 # 设置步长
    restore_data, s = samp_process(Beta=Beta, y=y, t=t, A=A)
    plt.figure(dpi=100)
    plt.stem(np.arange(m), restore_data, linefmt="r-", use_line_collection=True)
    plt.title("原始信号")
    plt.show()
