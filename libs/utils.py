import os
import threading
import torch
import numpy as np
from sympy.matrices import Matrix, GramSchmidt
import torch.nn.functional as F

from . import config


def cal_capacity(batch_x, snr):
    """计算系统容量"""
    if snr is None:
        snr = 100
    batch_x = batch_x.view(-1, 2, 32, 32)
    batch_x = torch.sqrt(torch.pow(batch_x[:, 0, :, :], 2) + torch.pow(batch_x[:, 1, :, :], 2))
    return torch.log2(torch.sum(1 + torch.linalg.svd(batch_x)[1] * snr / config.Nt, dim=-1)).mean().item()


def load_Phi(Phi_path, raw=None, col=None):
    """加载观测矩阵Phi"""
    if os.path.exists(Phi_path):
        Phi = np.load(Phi_path)
    else:
        Phi = orthogo_matrix(np.random.randn(raw, col))
        np.save(Phi_path, Phi)
    return Phi


def orthogo_matrix(x):
    """按行施密特单位正交化，x为numpy数组"""
    # 每一个行向量进行施密特正交化
    m, n = x.shape
    # 将numpy矩阵转为sympy的向量集合
    MA = [Matrix(col) for col in x]
    # 施密特正交化
    gram = GramSchmidt(MA)
    ort_list = []
    for i in range(m):
        vector = []
        for j in range(n):
            vector.append(float(gram[i][j]))  # 需要将sympy.core.numbers.Float转为float类型
        ort_list.append(vector)
    # 单位化
    ort_list = torch.tensor(ort_list)
    ort_list = F.normalize(ort_list, dim=1)
    return ort_list.numpy()


def normalized(x):
    """将数据 min-max 归一化"""
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def nmse(a, target, dtype):
    """计算一个batch的a和target的NMSE"""
    if dtype == "torch":
        power = target ** 2
        difference = (target - a) ** 2
        return (10 * torch.log10(difference.sum(dim=-1) / power.sum(dim=-1))).mean().item()
    elif dtype == "np":
        power = target ** 2
        difference = (target - a) ** 2
        return (10 * np.log10(difference.sum(axis=-1) / power.sum(axis=-1))).mean().item()


class SingletonType(type):
    """单例元类"""
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance


def res_unit(func, input_):
    """用过残差网络提取特征"""
    out = func(input_)
    output = out + input_  # 加入残差结构
    return output


def net_standardization(input_):
    """ 标准化处理"""
    mean = torch.mean(input_, dim=-1, keepdim=True)
    std = torch.std(input_, dim=-1, keepdim=True)
    output = (input_ - mean) / std
    return output


def gs_noise(x, snr):
    """
    对模型加入高斯白噪声

    噪音强度为: 10 ** (snr / 10)，其中snr为对应的信噪比
    注: 信号的幅度的平方==信号的功率，因此要开平方根
    :param x: 信号
    :param snr: 信噪比
    :return: 加入噪声后的结果
    """
    return x
    if snr is None:
        return x
    with torch.no_grad():
        x_power = (torch.sum(x ** 2) / x.numel()).item()  # 信号的功率
        noise_power = x_power / 10 ** (snr / 10)  # 噪声的功率
        gaussian = torch.normal(0, pow(noise_power, 0.5), x.size(), device=config.device)  # 产生对应信噪比的高斯白噪声
        return x + gaussian


def get_gs_noise_std(x, snr):
    """通过信噪比SNR，获取噪声方差"""
    if snr is None:
        return 0.001
    with torch.no_grad():
        x_power = (torch.sum(x ** 2) / x.numel()).item()  # 信号的功率
        noise_power = x_power / 10 ** (snr / 10)  # 噪声的功率
        return pow(noise_power, 0.5)


def obj_wrapper(func):
    """对函数进行封装，方便对象的调用"""

    def call_func(self, *args, **kwargs):
        return func(*args, **kwargs)

    return call_func


def criteria_loop_wrapper(func):
    """不同评价指标的装饰器"""

    def call_func(obj, *args, **kwargs):
        criteria_li = kwargs.get("criteria", config.criteria_list)
        for criteria in criteria_li:
            func(obj, criteria=criteria, *args, **kwargs)

    return call_func


def model_snr_loop_wrapper(func):
    """模型加入不同信噪比噪声的实例方法的装饰器"""

    def call_func(obj, *args, **kwargs):
        # obj 为该实例对象
        model_snrs = kwargs.get('model_snrs', config.model_SNRs)
        for model_snr in model_snrs:
            func(obj, model_snr=model_snr, *args, **kwargs)

    return call_func


def ratio_loop_wrapper(func):
    """执行不同压缩率的实例方法的装饰器"""

    def call_func(obj, *args, **kwargs):
        # obj 为该实例对象
        r_list = kwargs.get('r_list', config.ratio_list)
        for ratio in r_list:
            func(obj, ratio=ratio, *args, **kwargs)

    return call_func


def v_loop_wrapper(func):
    """执行不同速度的实例方法的装饰器"""

    def call_func(obj, *args, **kwargs):
        # obj 为该实例对象
        velocity = kwargs.get('v_list', config.velocity_list)
        for v in velocity:
            func(obj, v=v, *args, **kwargs)

    return call_func


def rec_mkdir(file_path):
    """
    以执行python程序的位置做为当前目录，递归创建文件夹，如果文件夹不存在，就创建，存在就跳过，继续下一层
    file_path的格式为: dir/dir/dir/filename

    """
    dir_list = file_path.split('/')[:-1]
    cur_dir = ""
    for dr in dir_list:
        cur_dir += dr
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        cur_dir += "/"
