# -*- coding: UTF-8 -*-
"""各类压缩感知算法的封装并集成在CSI反馈系统中"""
import h5py
import numpy as np
import torch.nn.functional as F
import torch
import pickle
import tqdm
import time


FUNC_METHOD = dict()  # 保存所有的重构方法与对应的重构函数，key：重构信号的方法，value：对应重构信号方法的函数


def route(method):
    """建立路由"""
    def decorator(func):
        """使用装饰器，生成路由函数字典"""
        FUNC_METHOD[method] = func

        def call_func(*args, **kwargs):
            return func(*args, **kwargs)

        return call_func
    return decorator


class CS(object):
    """
    对数据进行压缩感知重构，并评估重构结果，每次只能对一条数据进行测试，
    同时，每次测试只能在一种信噪比下使用一种CS的组合方法，进行测试


    算法：
        DCT: 离散余弦变换
        IDCT: 离散余弦逆变换
        FFT：快速傅里叶变换
        OMP：正交匹配追踪
        SAMP；自适应匹配追踪算法
    """

    def __init__(self, **kwargs):
        """
        参数与变量：
                y: 观测向量 [m, 1]
                data: 原始信号向量 [n, 1]
                s: data在稀疏基A下的稀疏表示
                k: 稀疏度，非零个数
                Beta: 传感矩阵
                snr: 信噪比
                sparse: 压缩方法
                restore: 重构方法
                A: 稀疏基矩阵
                Fi： 观测矩阵
                Fi_m：压缩比例
                t：步长


        初始化属性:
                self.sparse_method -----> config.method_list  # 确定稀疏基
                self.restore_method -----> config.method_list  # 确定重构算法
                self.snr ----> config.SNRs  # 得到信噪比
                self.full_sampling = full_sampling  # 设置是否全采样
                self.k = config.k  # 设置稀疏度
                self.Fi_m -----> cs_data_length / ratio  # 观得到测矩阵的行数
                self.t -----> config.t  # 设置SAMP算法步长
                self.Fi_ratio = config.cs_ratio_list  # 压缩率
                self.velocity = config.velocity  # 不同速度下的数据集
        """
        # 初始化实例属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.sparse_matrix = load_sparse_base(self.sparse_method.upper())  # 加载稀疏基

    def run(self):
        """主函数：对数据进行压缩、重构、计算相似度与损失，然后返回测试结果"""
        similarity_list = list()
        loss_list = list()
        time_list = list()
        bar = tqdm.tqdm(dataset(self.velocity))
        for data in bar:
            data = data.reshape(-1, 1)

            # 数据压缩
            start_time = time.time()  # 开始时间
            y, Beta, k = self.__compress(data)

            # 经过信道加噪
            y_add_noise = self.__channel(y)

            # 重构信号
            restore_data = self.__restore(self.restore_method, y_add_noise, Beta, k)
            stop_time = time.time()  # 结束时间

            # 计算相似度、损失、消耗的时间
            similarity, loss = self.__evaluate(data, restore_data)
            similarity_list.append(similarity.item())
            loss_list.append(loss.item())
            time_list.append(stop_time-start_time)

            # 显示进度
            bar.set_description("ratio:{}\t{}:{}\tnoise_SNR:{}dB\tloss:{:}\tsimilarity:{:.3f}"
                                .format(self.Fi_ratio, self.sparse_method, self.restore_method, self.snr, loss.item(), similarity.item()))
        # 将测试结果返回
        return self.__save_result(loss_list, similarity_list, time_list)

    def __compress(self, data):
        """
        压缩数据：
            获得稀疏基与观测矩阵Fi，通过计算，返回传感矩阵与观测向量,根据是否进行全采样，选择不同压缩方式
            is_full_sampling: 是否进行全采样
        :param data:[?, 1]
        :return: 传感矩阵beta，压缩后的数据y,稀疏度k
        """
        dim = data.shape[0]
        if not self.full_sampling:
            Fi = np.random.randn(self.Fi_m, dim)  # 确定观测矩阵大小
            y = np.matmul(Fi, data)  # 观测向量
            Beta = np.matmul(Fi, self.sparse_matrix)  # 传感矩阵
            # 确定稀疏度k
            k = self.k
        else:
            Beta = None
            k = None
            y = np.matmul(self.sparse_matrix, data)
            # 只保留前Fi_m个绝对值最大的元素,其余置为0, Fi_m = data_length / Fi_ratio
            temp_y = np.abs(y)
            temp_max = np.sort(temp_y, axis=0)[dim - self.Fi_m]
            y[temp_y < temp_max] = 0
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
            y_row, y_col = np.real(y).shape
            y_power_real = np.sum(np.real(y) ** 2) / np.real(y).size
            y_power_imag = np.sum(np.imag(y) ** 2) / np.imag(y).size
            noise_power_real = (y_power_real / (10 ** (self.snr / 10)))
            noise_power_imag = (y_power_imag / (10 ** (self.snr / 10)))
            gaussian = np.random.normal(0, np.sqrt(noise_power_real), (y_row, y_col)) + 1j * np.random.normal(0, np.sqrt(noise_power_imag), (y_row, y_col))
        else:
            y_power = (np.sum(y ** 2) / y.size)  # 信号的功率
            noise_power = (y_power / (10 ** (self.snr / 10)))  # 噪声的功率
            gaussian = np.random.normal(0, np.sqrt(noise_power), (y.shape[0], y.shape[1]))
        y_add_noise = y + gaussian
        return y_add_noise

    def __restore(self, method, y_add_noise, Beta, k):
        """
        对y进行重构，恢复成原始数据
        :param y_add_noise: 经过信道后的压缩数据
        :param Beta: 传感矩阵
        :return: restore_data:重构后的数据
        func不指向route中的call_func，指向被修饰的不同实例方法
        """
        func = FUNC_METHOD.get(method.lower(), None)
        if func:
            return func(self, y_add_noise, Beta, k)
        else:
            raise KeyError("重构方法错误")

    @staticmethod
    def __evaluate(data, refine_data):
        """评估余弦相似度、mse损失"""
        data = torch.tensor(data)
        refine_data = torch.tensor(refine_data)
        loss = F.mse_loss(data, refine_data)
        similarity = torch.cosine_similarity(data, refine_data, dim=0)
        return similarity, loss

    def __save_result(self, loss_list, similarity_list, time_list):
        """计算平局损失、平均相似度与平均计算时间"""
        avg_loss = np.mean(loss_list)
        avg_similarity = np.mean(similarity_list)
        avg_time = np.mean(time_list)

        # 保存结果到字典中并返回
        return {"snr": self.snr, "NMSE": avg_loss, "相似度": avg_similarity, "time": avg_time}

    @route("dct_omp")
    def __DCT_OMP(self, y_add_noise, Beta, k, *args):
        """
        当稀疏基是dct变换基时，将恢复的稀疏系数s计算idct，恢复数据data，*args仅用来接收多余参数，防止程序崩溃，无其他用途
        """
        restore_s = self.__OMP(y_add_noise, Beta, k)
        restore_data = np.matmul(self.sparse_matrix.T, restore_s)
        return restore_data

    @route("fft_omp")
    def __FFT_OMP(self, y_add_noise, Beta, k, *args):
        """
        当稀疏基是fft变换基时，实部虚部分开处理，进行重构,重构稀疏系数s的实部与虚部，
        重构出来的两部分合并成复数，计算ifft，恢复数据data，*args仅用来接收多余参数，防止程序崩溃，无其他用途
        """
        Beta_real = np.real(Beta)
        Beta_imag = np.imag(Beta)
        restore_s_real = self.__OMP(y_add_noise, Beta_real, k)  # s实部
        restore_s_imag = self.__OMP(y_add_noise, Beta_imag, k)  # s虚部
        restore_s = restore_s_real + restore_s_imag * 1j
        restore_data = np.matmul(self.sparse_matrix.T, restore_s)  # IFFT变换，恢复data
        return np.real(restore_data)  # 取实部，作为恢复的信号

    @route("dct_samp")
    def __DCT_SAMP(self, y_add_noise, Beta, *args):
        """基于DCT稀疏基的SAMP重构算法，*args仅用来接收多余参数，防止程序崩溃，无其他用途"""
        s = self.__SAMP(y_add_noise, Beta, self.t)
        return np.matmul(self.sparse_matrix.T, s)

    @route("fft_samp")
    def __FFT_SAMP(self, y_add_noise, Beta, *args):
        """基于FFT稀疏基的SAMP重构算法，*args仅用来接收多余参数，防止程序崩溃，无其他用途"""
        Beta_real = np.real(Beta)
        Beta_imag = np.imag(Beta)
        restore_s_real = self.__SAMP(y_add_noise, Beta_real, self.t)  # s实部
        restore_s_imag = self.__SAMP(y_add_noise, Beta_imag, self.t)  # s虚部
        restore_s = restore_s_real + restore_s_imag * 1j
        restore_data = np.matmul(self.sparse_matrix.T, restore_s)  # IFFT变换，恢复data
        return np.real(restore_data)  # 取实部，作为恢复的信号

    @route("dct_sp")
    def __DCT_SP(self, y_add_noise, Beta, k, *args):
        """基于DCT稀疏基的SP重构算法，*args仅用来接收多余参数，防止程序崩溃，无其他用途"""
        s = self.__SP(y_add_noise, Beta, k)
        return np.matmul(self.sparse_matrix.T, s)

    @route("fft_sp")
    def __FFT_SP(self, y_add_noise, Beta, k, *args):
        """基于FFT稀疏基的SP重构算法，*args仅用来接收多余参数，防止程序崩溃，无其他用途"""
        Beta_real = np.real(Beta)
        Beta_imag = np.imag(Beta)
        restore_s_real = self.__SP(y_add_noise, Beta_real, k)  # s实部
        restore_s_imag = self.__SP(y_add_noise, Beta_imag, k)  # s虚部
        restore_s = restore_s_real + restore_s_imag * 1j
        restore_data = np.matmul(self.sparse_matrix.T, restore_s)  # IFFT变换，恢复data
        return np.real(restore_data)  # 取实部，作为恢复的信号

    @route("idct")
    def __IDCT(self, y_add_noise, *args):
        """DCT逆变换，*args仅用来接收多余参数，防止程序崩溃，无其他用途"""
        return np.matmul(self.sparse_matrix.T, y_add_noise)

    @route("ifft")
    def __IFFT(self, y_add_noise, *args):
        """FFT逆变换，*args仅用来接收多余参数，防止程序崩溃，无其他用途"""
        x_row = y_add_noise.shape[0]
        S_conj = np.conjugate(self.sparse_matrix)
        data = np.matmul(S_conj, y_add_noise) / x_row
        return np.real(data)  # 取实部，作为恢复的信号

    @staticmethod
    def __OMP(y, Beta, k):
        """
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

    @staticmethod
    def __SAMP(y, Beta, t):
        """
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

    @staticmethod
    def __SP(y, Beta, k):
        """
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

    def __call__(self, *args, **kwargs):
        return self.run()


def dataset(velocity):
    """加载cs测试的数据集"""
    current_path = ".." if __name__ == "__main__" else "."
    data_path = r"{}/data/matlab/test_100_32_{}_H.mat".format(current_path, velocity)
    return h5py.File(data_path)["save_H"]


def load_sparse_base(method):
    if method == "DCT":
        return pickle.load(open("./data/dct_32.pkl", "rb"))  # 获得稀疏基
    elif method == "FFT":
        return pickle.load(open("./data/fft_32.pkl", "rb"))  # 获得稀疏基
    elif method == "unit":
        return np.identity(32)
    else:
        exit("压缩方法错误")


if __name__ == '__main__':
    print(dataset(150).shape)
