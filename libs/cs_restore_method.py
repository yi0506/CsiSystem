import numpy as np
import torch
import pickle


def OMP(y, Beta, k):
    """
    OMP算法
    :param y: 观测向量 [m, 1]
    :param Beta: 传感矩阵 [m, n], m << n
    :param k: 稀疏度
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
        Beta[:, idx] = np.zeros((Beta_row,))

        # 求s的最小二乘解,所求结果为s中不为0的位置处的值
        s_ls = np.matmul(np.linalg.pinv(Beta_new[:, :i + 1]), y)

        # 更新残差
        residual_error = y - np.matmul(Beta_new[:, :i + 1], s_ls)

    # 得到data的稀疏系数表示s
    s[Beta_idx.flatten()] = s_ls
    return s


def SAMP(y, Beta, t):
    """
    SAMP算法
    :param y: 观测向量 [m, 1]
    :param Beta: 传感矩阵 [m, n]
    :param t: 步长
    :return: 重构稀疏系数 s[n, 1]
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

        # y = Beta_t * s，求s的最小二乘解(Least Square)
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
            if i + 1 == iter_num:  # 最后一次循环
                Beta_idx = F  # 更新Beta_idx与s_ls匹配，防止报错
        else:
            Beta_idx = F
            res_error = res_new

    # 得到s的稀疏系数
    s[Beta_idx.flatten()] = s_ls
    return s


def SP(y, Beta, k):
    """
    SP算法
    :param y: 观测向量 [m, 1]
    :param Beta: 传感矩阵 [m, n], m << n
    :param k: 稀疏度
    :return: 重构稀疏系数 s[n, 1]
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
    return s


def generate_dct_sparse_base(dim):
    """获得DCT稀疏基"""
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


def generate_fft_sparse_base(N):
    """
    获得FFT稀疏基

    X[k] = $\sum_{n=0}^(N-1)x[n]*exp(-j*2*pi*k*n/N)$
    k: frequency index
    N: length of complex sinusoid in samples
    n: 当前样本
    """
    n = np.arange(N).reshape(1, N)
    m = n.T * n / N  # [N, 1] * [1, N] = [N, N]
    S = np.exp(-1j * 2 * np.pi * m)
    # 保存稀疏基
    pickle.dump(S, open("../data/fft_32.pkl", "wb"))