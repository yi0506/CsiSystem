import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.functional import cosine_similarity

import config
from utils import obj_wrapper, nmse


@obj_wrapper
def td_fista_test(model, Qinit, data_loader, snr, info: str = ""):
    """ISTANet 评估模型"""
    model.to(config.device).eval()
    # 测试模型在某个信噪比下的效果，作为模型的评价效果
    nmse_list = list()
    similarity_list = list()
    capacity_list = list()
    loss_list = list()
    for _, target in enumerate(data_loader):
        with torch.no_grad():
            target = target.to(config.device)
            [output, _, _] = model(target, Qinit, snr)
            target = target - 0.5
            output = output - 0.5
            cur_similarity = cosine_similarity(output, target, dim=-1).mean().cpu().item()
            cur_loss = F.mse_loss(output, target).item()
            cur_nmse = nmse(output, target, "torch")
            # cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item()  # 信道容量:SVD分解SVD分解
            cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * 10 / config.Nt)).item()  # 信道容量:SVD分解SVD分解
            capacity_list.append(cur_capacity / target.size()[0])
            nmse_list.append(cur_nmse)
            loss_list.append(cur_loss)
            similarity_list.append(cur_similarity)

    # 计算平均相似度与损失
    avg_nmse = np.mean(nmse_list)
    avg_similarity = np.mean(similarity_list)
    avg_capacity = np.mean(capacity_list)
    avg_loss = np.mean(loss_list)
    print(info + "\tSNR:{}dB\tloss:{:.5e}\tnmse:{:.3f}\tsimilarity:{}\tcapacity{}".format(snr, avg_loss, avg_nmse, avg_similarity, avg_capacity))
    return {"相似度": avg_similarity, "NMSE": avg_nmse}


@obj_wrapper
def fista_test(model, Phi, Qinit, data_loader, snr, info: str = ""):
    """FISTANet 评估模型"""
    model.to(config.device).eval()
    # 测试模型在某个信噪比下的效果，作为模型的评价效果
    nmse_list = list()
    similarity_list = list()
    capacity_list = list()
    loss_list = list()
    for _, target in enumerate(data_loader):
        with torch.no_grad():
            target = target.to(config.device)
            Phix = torch.mm(target, torch.transpose(Phi, 0, 1))
            [output, _, _] = model(Phix, Phi, Qinit, snr)
            target = target - 0.5
            output = output - 0.5
            cur_similarity = cosine_similarity(output, target, dim=-1).mean().cpu().item()
            cur_loss = F.mse_loss(output, target).item()
            cur_nmse = nmse(output, target, "torch")
            # cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item()  # 信道容量:SVD分解SVD分解
            cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * 10 / config.Nt)).item()  # 信道容量:SVD分解SVD分解
            capacity_list.append(cur_capacity / target.size()[0])
            nmse_list.append(cur_nmse)
            loss_list.append(cur_loss)
            similarity_list.append(cur_similarity)

    # 计算平均相似度与损失
    avg_nmse = np.mean(nmse_list)
    avg_similarity = np.mean(similarity_list)
    avg_capacity = np.mean(capacity_list)
    avg_loss = np.mean(loss_list)
    print(info + "\tSNR:{}dB\tloss:{:.5e}\tnmse:{:.3f}\tsimilarity:{}\tcapacity{}".format(snr, avg_loss, avg_nmse, avg_similarity, avg_capacity))
    return {"相似度": avg_similarity, "NMSE": avg_nmse}


@obj_wrapper
def ista_test(model, Phi, Qinit, data_loader, snr, info: str = ""):
    """ISTANet 评估模型"""
    model.to(config.device).eval()
    # 测试模型在某个信噪比下的效果，作为模型的评价效果
    nmse_list = list()
    similarity_list = list()
    capacity_list = list()
    loss_list = list()
    for _, target in enumerate(data_loader):
        with torch.no_grad():
            target = target.to(config.device)
            Phix = torch.mm(target, torch.transpose(Phi, 0, 1))
            [output, _] = model(Phix, Phi, Qinit, snr)
            target = target - 0.5
            output = output - 0.5
            cur_similarity = cosine_similarity(output, target, dim=-1).mean().cpu().item()
            cur_loss = F.mse_loss(output, target).item()
            cur_nmse = nmse(output, target, "torch")
            # cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item()  # 信道容量:SVD分解SVD分解
            cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * 10 / config.Nt)).item()  # 信道容量:SVD分解SVD分解
            capacity_list.append(cur_capacity / target.size()[0])
            nmse_list.append(cur_nmse)
            loss_list.append(cur_loss)
            similarity_list.append(cur_similarity)

    # 计算平均相似度与损失
    avg_nmse = np.mean(nmse_list)
    avg_similarity = np.mean(similarity_list)
    avg_capacity = np.mean(capacity_list)
    avg_loss = np.mean(loss_list)
    print(info + "\tSNR:{}dB\tloss:{:.5e}\tnmse:{:.3f}\tsimilarity:{}\tcapacity{}".format(snr, avg_loss, avg_nmse, avg_similarity, avg_capacity))
    return {"相似度": avg_similarity, "NMSE": avg_nmse}


@obj_wrapper
def comm_csi_test(model, data_loader, snr, info: str = ""):
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
    capacity_list = list()
    loss_list = list()
    for _, input_ in enumerate(data_loader):
        with torch.no_grad():
            input_ = input_.to(config.device)
            output = model(input_, snr)
            input_ = input_ - 0.5
            output = output - 0.5
            cur_similarity = cosine_similarity(output, input_, dim=-1).mean().cpu().item()
            cur_loss = F.mse_loss(output, input_).item()
            cur_nmse = nmse(output, input_, "torch")
            # cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item()  # 信道容量:SVD分解
            cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * 10 / config.Nt)).item()  # 信道容量:SVD分解SVD分解
            capacity_list.append(cur_capacity / input_.size()[0])
            nmse_list.append(cur_nmse)
            loss_list.append(cur_loss)
            similarity_list.append(cur_similarity)

    # 计算平均相似度与损失
    avg_nmse = np.mean(nmse_list)
    avg_similarity = np.mean(similarity_list)
    avg_capacity = np.mean(capacity_list)
    avg_loss = np.mean(loss_list)
    print(info + "\tSNR:{}dB\tloss:{:.5e}\tnmse:{:.3f}\tsimilarity:{}\tcapacity:{}".format(snr, avg_loss, avg_nmse, avg_similarity, avg_capacity))
    return {"相似度": avg_similarity, "NMSE": avg_nmse}


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
    capacity_list = list()
    loss_list = list()
    for _, input_ in enumerate(data_loader):
        with torch.no_grad():
            input_ = input_.to(config.device)
            output = model(input_, snr)
            input_ = input_ - 0.5
            output = output - 0.5
            cur_similarity = cosine_similarity(output, input_, dim=-1).mean().cpu().item()
            cur_loss = F.mse_loss(output, input_).item()
            cur_nmse = nmse(output, input_, "torch")
            # cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item()  # 信道容量:SVD分解
            cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * 10 / config.Nt)).item()  # 信道容量:SVD分解SVD分解
            capacity_list.append(cur_capacity / input_.size()[0])
            nmse_list.append(cur_nmse)
            loss_list.append(cur_loss)
            similarity_list.append(cur_similarity)

    # 计算平均相似度与损失
    avg_nmse = np.mean(nmse_list)
    avg_similarity = np.mean(similarity_list)
    avg_capacity = np.mean(capacity_list)
    avg_loss = np.mean(loss_list)
    print(info + "\tSNR:{}dB\tloss:{:.5e}\tnmse:{:.3f}\tsimilarity:{}\tcapacity{}".format(snr, avg_loss, avg_nmse, avg_similarity, avg_capacity))
    return {"相似度": avg_similarity, "NMSE": avg_nmse}


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
    capacity_list = list()
    loss_list = list()
    for _, (target, y) in enumerate(data_loader):
        with torch.no_grad():
            target = target.to(config.device)
            y = y.to(config.device)
            output = model(y, snr)
            target = target - 0.5
            output = output - 0.5
            cur_similarity = cosine_similarity(output, target, dim=-1).mean().cpu().item()
            cur_nmse = nmse(output, target, "torch")
            cur_loss = F.mse_loss(output, target).item()
            # cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item()  # 信道容量:SVD分解
            cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * 10 / config.Nt)).item()  # 信道容量:SVD分解SVD分解
            capacity_list.append(cur_capacity / y.size()[0])
            nmse_list.append(cur_nmse)
            loss_list.append(cur_loss)
            similarity_list.append(cur_similarity)

    # 计算平均相似度与损失
    nmse_loss = np.mean(nmse_list)
    avg_similarity = np.mean(similarity_list)
    avg_nmse = np.mean(nmse_list)
    avg_capacity = np.mean(capacity_list)
    avg_loss = np.mean(loss_list)
    print(info + "\tSNR:{}dB\tloss:{:.5e}\tnmse:{:.3f}\tsimilarity:{}\tcapacity{}".format(snr, avg_loss, avg_nmse, avg_similarity, avg_capacity))
    return {"相似度": avg_similarity, "NMSE": nmse_loss}
