import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import time

import config
from utils import rec_mkdir, obj_wrapper, nmse


@obj_wrapper
def test(model, data_loader, snr, info: str = ""):
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
    nmse_list = list()
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
            cur_nmse = nmse(output, input_, "torch")
            cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item()  # 信道容量:SVD分解
            capacity_list.append(cur_capacity / input_.size()[0])
            nmse_list.append(cur_nmse / input_.size()[0])
            similarity_list.append(cur_similarity)
            time_list.append((stop_time - start_time) / input_.size()[0])

    # 计算平均相似度与损失
    avg_nmse = np.mean(nmse_list)
    avg_similarity = np.mean(similarity_list)
    avg_time = np.mean(time_list)
    avg_capacity = np.mean(capacity_list)
    print(info + "\tSNR:{}dB\tnmse:{:.3f}\tsimilarity:{:.3f}\ttime:{:.4f}\tcapacity:{:.4f}".format(snr, avg_nmse, avg_similarity, avg_time, avg_capacity))
    return {"相似度": avg_similarity, "NMSE": avg_nmse, "time": avg_time, "Capacity": avg_capacity}


@obj_wrapper
def csp_test(model, data_loader, snr, info: str = ""):
    """
    评估模型，并返回结果

    model: 模型
    model_path: 模型的加载路径
    data_loader: 数据集迭代器
    snr: 加入噪声的信噪比
    info: 额外的结果描述信息

    """
    model.to(config.device).eval()
    # 测试模型在某个信噪比下的效果，作为模型的评价效果
    nmse_list = list()
    similarity_list = list()
    time_list = list()
    capacity_list = list()
    for _, (target, y) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            target = target.to(config.device)
            y = y.to(config.device)
            start_time = time.time()
            output = model(y, snr)
            stop_time = time.time()
            cur_similarity = cosine_similarity(output, target, dim=-1).mean().cpu().item()
            cur_nmse = nmse(output, target, "torch")
            cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item()  # 信道容量:SVD分解
            capacity_list.append(cur_capacity / y.size()[0])
            nmse_list.append(cur_nmse / y.size()[0])
            similarity_list.append(cur_similarity)
            time_list.append((stop_time - start_time) / y.size()[0])

    # 计算平均相似度与损失
    nmse_loss = np.mean(nmse_list)
    avg_similarity = np.mean(similarity_list)
    avg_time = np.mean(time_list)
    avg_capacity = np.mean(capacity_list)
    print(info + "\tSNR:{}dB\tnmse:{:.3f}\tsimilarity:{:.3f}\ttime:{:.4f}\tcapacity:{:.4f}".format(snr, nmse_loss, avg_similarity, avg_time, avg_capacity))
    return {"相似度": avg_similarity, "NMSE": nmse_loss, "time": avg_time, "Capacity": avg_capacity}


@obj_wrapper
def csp_train(model, epoch, save_path, data_loader, info):
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
        for idx, (target, y) in enumerate(bar):
            optimizer.zero_grad()
            target = target.to(config.device)
            y = y.to(config.device)
            output = model(y, None)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            bar.set_description(info + "\tepoch:{}\tidx:{}\tloss:{:}".format(i + 1, idx, loss.item()))
            if loss.item() < init_loss:
                init_loss = loss.item()
                rec_mkdir(save_path)  # 保证该路径下文件夹存在
                torch.save(model.state_dict(), save_path)
            if loss.item() < 1e-6:
                return


@obj_wrapper
def train(model, epoch, save_path, data_loader, info):
    """
    进行模型训练

    model: 模型
    epoch: 模型迭代次数
    save_path: 模型保存路径
    data_loader: 数据集迭代器
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
            output = model(data, None)
            loss = F.mse_loss(output, data)  # 计算损失
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()  # 梯度更新
            bar.set_description(info + "\tepoch:{}\tidx:{}\tloss:{:}".format(i + 1, idx, loss.item()))
            if loss.item() < init_loss:
                init_loss = loss.item()
                rec_mkdir(save_path)  # 保证该路径下文件夹存在
                torch.save(model.state_dict(), save_path)
            if loss.item() < 1e-6:
                return


@obj_wrapper
def train_stu(teacher, stu, epoch, save_path, data_loader, info):
    """
    进行学生模型训练

    teacher: 教师模型
    stu: 教师模型
    epoch: 模型迭代次数
    save_path: 模型保存路径
    data_loader: 数据集迭代器
    info: 额外的结果描述信息
    """
    teacher.to(config.device).eval()
    stu.to(config.device).train()
    optimizer = Adam(stu.parameters())
    init_loss = 1
    for i in range(epoch):
        bar = tqdm(data_loader)
        for idx, data in enumerate(bar):
            optimizer.zero_grad()
            data = data.to(config.device)
            teacher_ouput = teacher(data, None)
            stu_output = stu(data, None)
            loss = F.mse_loss(stu_output, teacher_ouput)
            loss.backward()
            optimizer.step()
            bar.set_description(info + "\tepoch:{}\tidx:{}\tloss:{:}".format(i + 1, idx, loss.item()))
            if loss.item() < init_loss:
                init_loss = loss.item()
                rec_mkdir(save_path)  # 保证该路径下文件夹存在
                torch.save(stu.state_dict(), save_path)
            if loss.item() < 1e-6:
                return
