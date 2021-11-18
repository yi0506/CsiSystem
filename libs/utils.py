import os
import numpy as np
import pickle
import threading
import torch.nn.functional as F
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, mse_loss
from tqdm import tqdm
import time

from libs import config


class SingletonType(type):
    """单例元类"""
    _instance_lock = threading.Lock()
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType,cls).__call__(*args, **kwargs)
        return cls._instance
    
    
def res_unit(func, input_):
        """用过残差网络提取特征"""
        out = func(input_)
        output = out + input_  # 加入残差结构
        return output


def net_normalize(input_):
    """ 标准化处理"""
    mean = torch.mean(input_, dim=-1, keepdim=True)
    std = torch.std(input_, dim=-1, keepdim=True)
    output = (input_ - mean) / std
    return output
    
    
def gs_noise(x, snr):
    """
    对模型加入高斯白噪声
    
    噪音强度为：10 ** (snr / 10)，其中snr为对应的信噪比
    注：信号的幅度的平方==信号的功率，因此要开平方根
    :param x: 信号
    :param snr: 信噪比
    :return: 加入噪声后的结果
    """
    if snr is None:
        return x
    with torch.no_grad():
        x_power = (torch.sum(x ** 2) / x.numel())  # 信号的功率
        noise_power = (x_power / torch.tensor(10 ** (snr / 10)))  # 噪声的功率
        gaussian = torch.normal(0, torch.sqrt(noise_power).item(), x.size(), device=config.device)  # 产生对应信噪比的高斯白噪声
        return x + gaussian


def test(model, data_loader, snr, info: str=""):
    """
    评估模型，并返回结果
    
    model: 模型
    model_path: 模型的加载路径
    data_loader: 数据集迭代器
    snr: 加入噪声的信噪比
    info: 额外的结果描述信息
    model_snr: 加入某种信噪比噪声的情况下训练好的模型
    
    """
    model.to(config.device).eval()
    # 测试模型在某个信噪比下的效果，作为模型的评价效果
    loss_list = list()
    similarity_list = list()
    time_list = list()
    capacity_list = list()
    for _, input_ in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            input_ = input_.to(config.device)
            start_time = time.time()
            output = model(input_, snr)
            stop_time = time.time()
            cur_similarity = cosine_similarity(output, input_, dim=-1).mean().cpu().item()
            cur_loss = mse_loss(output, input_).cpu().item()
            cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item()  # 信道容量:SVD分解
            capacity_list.append(cur_capacity / input_.size[0])
            loss_list.append(cur_loss / input_.size[0])
            similarity_list.append(cur_similarity)
            time_list.append((stop_time - start_time) / input_.size[0])

    # 计算平均相似度与损失
    avg_loss = np.mean(loss_list)
    avg_similarity = np.mean(similarity_list)
    avg_time = np.mean(time_list)
    avg_capacity = np.mean(capacity_list)
    print(info + "\tSNR:{}dB\tloss:{:.3f}\tsimilarity:{:.3f}\ttime:{:.4f}\tcapacity:{:.4f}".format(snr, avg_loss, avg_similarity, avg_time, avg_capacity))
    return {"相似度": avg_similarity, "NMSE": avg_loss, "time": avg_time, "Capacity": avg_capacity}


def train(model, epoch, save_path, data_loader, model_snr, info):
    """
    进行模型训练
    
    model: 模型
    epoch: 模型迭代次数
    save_path: 模型保存路径
    data_loader: 数据集迭代器
    model_snr: 对模型训练时，加入某种信噪比的噪声，train中的snr对应test中的model_snr
    info: 额外的结果描述信息
    """
    model.to(config.device).train()
    optimizer = Adam(model.parameters())
    init_loss = 1
    for i in range(epoch):
        bar = tqdm(data_loader)
        for idx, data in enumerate(bar):
            optimizer.zero_grad()  # 梯度置为零
            data = data.to(config.device)  # 转到GPU训练
            output = model(data, model_snr)
            similarity = torch.cosine_similarity(output, data, dim=-1).mean()  # 当前一个batch的相似度
            loss = F.mse_loss(output, data)  # 计算损失
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()  # 梯度更新
            bar.set_description(info + "\tSNR:{}dB\tepoch:{}\tidx:{}\tloss:{:}\tsimilarity:{:.3f}".format(model_snr, i + 1, idx, loss.item(), similarity.item()))
            if loss.item() < init_loss:
                init_loss = loss.item()
                rec_mkdir(save_path)  # 保证该路径下文件夹存在
                torch.save(model.state_dict(), save_path)
            if loss.item() < 1e-6:
                return


def model_snr_loop_wrapper(func):
    """模型加入不同信噪比噪声的实例方法的装饰器"""
    def call_func(obj, **kwargs):
        # obj 为该实例对象
        model_snrs = kwargs.get('model_snrs', config.model_SNRs)
        kwargs.pop('snr_list')
        for model_snr in model_snrs:
            func(obj, snr=model_snr, **kwargs)
    return call_func


def ratio_loop_wrapper(func):
    """执行不同压缩率的实例方法的装饰器"""
    def call_func(obj, **kwargs):
        # obj 为该实例对象
        r_list = kwargs.get('r_list', config.ratio_list)
        kwargs.pop('r_list')
        for ratio in r_list:
            func(obj, ratio=ratio, **kwargs)
    return call_func
    
    
def v_loop_wrapper(func):
    """执行不同速度的实例方法的装饰器"""
    def call_func(obj, **kwargs):
        # obj 为该实例对象
        velocity = kwargs.get('v_list', config.velocity_list)
        kwargs.pop('v_list')
        for v in velocity:
            func(obj, velocity=v, **kwargs)
    return call_func
    
    
def rec_mkdir(file_path):
    """
    以执行python程序的位置做为当前目录，递归创建文件夹，如果文件夹不存在，就创建，存在就跳过，继续下一层
    file_path的格式为：dir/dir/dir/filename
    
    """
    dir_list = file_path.split('/')[:-1]
    cur_dir = ""
    for dir in dir_list:
        cur_dir += dir
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        cur_dir += "/"
        
