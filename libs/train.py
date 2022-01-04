import torch.nn.functional as F
from torch.optim import Adam
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from utils import rec_mkdir, obj_wrapper
from csi_dataset import COMM_ValDataset


comm_val_dataloader = COMM_ValDataset().get_data_loader()


@obj_wrapper
def fista_train(model, epoch, Qinit, Phi, layer_num, save_path, data_loader, info):
    """
    进行模型训练

    model: 模型
    epoch: 模型迭代次数
    Qinit: 初始化参数
    Phi: 观测矩阵
    layer_num: 迭代次数
    save_path: 模型保存路径
    data_loader: 数据集迭代器
    model_snr: 对模型训练时，加入某种信噪比的噪声，train中的snr对应test中的model_snr
    info: 额外的结果描述信息
    """
    model.to(config.device).train()
    optimizer = Adam(model.parameters())
    init_loss = 1
    for i in range(epoch):
        torch.cuda.empty_cache()  # 清空缓存
        bar = tqdm(data_loader)
        for idx, batch_x in enumerate(bar):
            batch_x = batch_x.to(config.device)
            Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))  # 计算y
            [x_output, h_iter, loss_layers_sym] = model(Phix, Phi, Qinit)
            # 计算损失
            loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
            loss_constraint = torch.tensor(0).float().to(config.device)
            for k in range(layer_num):
                loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))
            loss_iteration = torch.tensor(0).float().to(config.device)
            for h_k in h_iter:
                loss_iteration += torch.mean(torch.pow(h_k - batch_x, 2))
            gamma = torch.Tensor([0.01]).to(config.device)
            miu = torch.Tensor([0.01]).to(config.device)
            loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint) + torch.mul(miu, loss_iteration)
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()
            bar.set_description(info + "\tepoch:{}\tidx:{}\tTotal Loss:{:.4e}\tDiscrepancy Loss:{:.4e}\tConstraint Loss{:.4e}\tIteration Loss{:.4e}".format(i + 1, idx, loss_all.item(), loss_discrepancy.item(), loss_constraint.item(), loss_iteration.item()))
            # 模型验证
            if (idx + 1) % 10 == 0:
                with torch.no_grad():
                    loss_list = list()
                    for idx, batch_x in enumerate(comm_val_dataloader):
                        batch_x = batch_x.to(config.device)
                        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))  # 计算y
                        [output, _, _] = model(Phix, Phi, Qinit)
                        batch_x = batch_x - 0.5
                        output = output - 0.5
                        cur_loss = F.mse_loss(output, batch_x).item()
                        loss_list.append(cur_loss)
                    val_loss = torch.mean(torch.tensor(loss_list))
                    if val_loss.item() < init_loss:
                        init_loss = val_loss.item()
                        rec_mkdir(save_path)  # 保证该路径下文件夹存在
                        torch.save(model.state_dict(), save_path)
                        print("保存模型:{}....val_loss:{:.4e}".format(save_path, init_loss))
                    if val_loss.item() < 1e-7:
                        return


@obj_wrapper
def td_fista_train(model, epoch, Qinit, layer_num, save_path, data_loader, info):
    """
    进行模型训练

    model: 模型
    epoch: 模型迭代次数
    Qinit: 初始化参数
    layer_num: 迭代次数
    save_path: 模型保存路径
    data_loader: 数据集迭代器
    model_snr: 对模型训练时，加入某种信噪比的噪声，train中的snr对应test中的model_snr
    info: 额外的结果描述信息
    """
    model.to(config.device).train()
    optimizer = Adam(model.parameters())
    init_loss = 1
    for i in range(epoch):
        torch.cuda.empty_cache()  # 清空缓存
        bar = tqdm(data_loader)
        for idx, batch_x in enumerate(bar):
            batch_x = batch_x.to(config.device)
            [x_output, h_iter, loss_layers_sym] = model(batch_x, Qinit)
            # 计算损失
            loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
            loss_constraint = torch.tensor(0).float().to(config.device)
            for k in range(layer_num):
                loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))
            loss_iteration = torch.tensor(0).float().to(config.device)
            for h_k in h_iter:
                loss_iteration += torch.mean(torch.pow(h_k - batch_x, 2))
            gamma = torch.Tensor([0.01]).to(config.device)
            miu = torch.Tensor([0.01]).to(config.device)
            loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint) + torch.mul(miu, loss_iteration)
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()
            bar.set_description(info + "\tepoch:{}\tidx:{}\tTotal Loss:{:.4e}\tDiscrepancy Loss:{:.4e}\tConstraint Loss{:.4e}\tIteration Loss{:.4e}".format(i + 1, idx, loss_all.item(), loss_discrepancy.item(), loss_constraint.item(), loss_iteration.item()))

            # 模型验证
            if (idx + 1) % 10 == 0:
                with torch.no_grad():
                    loss_list = list()
                    for idx, batch_x in enumerate(comm_val_dataloader):
                        batch_x = batch_x.to(config.device)
                        [output, _, _] = model(batch_x, Qinit)
                        batch_x = batch_x - 0.5
                        output = output - 0.5
                        cur_loss = F.mse_loss(output, batch_x).item()
                        loss_list.append(cur_loss)
                    val_loss = torch.mean(torch.tensor(loss_list))
                    if val_loss.item() < init_loss:
                        init_loss = val_loss.item()
                        rec_mkdir(save_path)  # 保证该路径下文件夹存在
                        torch.save(model.state_dict(), save_path)
                        print("保存模型:{}....val_loss:{:.4e}".format(save_path, init_loss))
                    if val_loss.item() < 1e-7:
                        return


@obj_wrapper
def ista_train(model, epoch, Qinit, Phi, layer_num, save_path, data_loader, info):
    """
    进行模型训练

    model: 模型
    epoch: 模型迭代次数
    Qinit: 初始化参数
    Phi: 观测矩阵
    layer_num: 迭代次数
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
        for idx, batch_x in enumerate(bar):
            batch_x = batch_x.to(config.device)
            Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))  # 计算y
            [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)
            # 计算损失
            loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
            loss_constraint = torch.tensor(0).float().to(config.device)
            for k in range(layer_num):
                loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))
            gamma = torch.Tensor([0.01]).to(config.device)
            loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()
            bar.set_description(info + "\tepoch:{}\tidx:{}\tTotal Loss:{:.4e}\tDiscrepancy Loss:{:.4e}\tConstraint Loss{:.4e}\t".format(i + 1, idx, loss_all.item(), loss_discrepancy.item(), loss_constraint.item()))

            # 模型验证
            if (idx + 1) % 10 == 0:
                with torch.no_grad():
                    loss_list = list()
                    for idx, batch_x in enumerate(comm_val_dataloader):
                        batch_x = batch_x.to(config.device)
                        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))  # 计算y
                        [output, _] = model(Phix, Phi, Qinit)
                        batch_x = batch_x - 0.5
                        output = output - 0.5
                        cur_loss = F.mse_loss(output, batch_x).item()
                        loss_list.append(cur_loss)
                    val_loss = torch.mean(torch.tensor(loss_list))
                    if val_loss.item() < init_loss:
                        init_loss = val_loss.item()
                        rec_mkdir(save_path)  # 保证该路径下文件夹存在
                        torch.save(model.state_dict(), save_path)
                        print("保存模型:{}....val_loss:{:.4e}".format(save_path, init_loss))
                    if val_loss.item() < 1e-7:
                        return


@obj_wrapper
def csp_train(model, epoch, Phi, save_path, data_loader, info):
    """
    进行模型训练

    model: 模型
    epoch: 模型迭代次数
    Phi: 观测矩阵
    save_path: 模型保存路径
    data_loader: 数据集迭代器
    model_snr: 对模型训练时，加入某种信噪比的噪声，train中的snr对应test中的model_snr
    info: 额外的结果描述信息
    """
    model.to(config.device).train()
    optimizer = Adam(model.parameters())
    init_loss = 1
    for i in range(epoch):
        torch.cuda.empty_cache()  # 清空缓存
        bar = tqdm(data_loader)
        for idx, target in enumerate(bar):
            optimizer.zero_grad()
            target = target.to(config.device)
            y = torch.mm(Phi, target.t()).t()  # ((m, dim) * (dim, batch)).T
            output = model(y, None)
            loss = F.mse_loss(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()
            bar.set_description(info + "\tepoch:{}\tidx:{}\tloss:{:.4e}".format(i + 1, idx, loss.item()))
            # 模型验证
            if (idx + 1) % 10 == 0:
                with torch.no_grad():
                    loss_list = list()
                    for idx, batch_x in enumerate(comm_val_dataloader):
                        batch_x = batch_x.to(config.device)
                        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))  # 计算y
                        output = model(Phix, None)
                        batch_x = batch_x - 0.5
                        output = output - 0.5
                        cur_loss = F.mse_loss(output, batch_x).item()
                        loss_list.append(cur_loss)
                    val_loss = torch.mean(torch.tensor(loss_list))
                    if val_loss.item() < init_loss:
                        init_loss = val_loss.item()
                        rec_mkdir(save_path)  # 保证该路径下文件夹存在
                        torch.save(model.state_dict(), save_path)
                        print("保存模型:{}....val_loss:{:.4e}".format(save_path, init_loss))
                    if val_loss.item() < 1e-7:
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
        torch.cuda.empty_cache()  # 清空缓存
        bar = tqdm(data_loader)
        for idx, data in enumerate(bar):
            optimizer.zero_grad()  # 梯度置为零
            data = data.to(config.device)  # 转到GPU训练
            output = model(data, None)
            loss = F.mse_loss(output, data)  # 计算损失
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()  # 梯度更新
            bar.set_description(info + "\tepoch:{}\tidx:{}\tloss:{:.4e}".format(i + 1, idx, loss.item()))
            # 模型验证
            if (idx + 1) % 10 == 0:
                with torch.no_grad():
                    loss_list = list()
                    for idx, batch_x in enumerate(comm_val_dataloader):
                        batch_x = batch_x.to(config.device)
                        output = model(batch_x, None)
                        batch_x = batch_x - 0.5
                        output = output - 0.5
                        cur_loss = F.mse_loss(output, batch_x).item()
                        loss_list.append(cur_loss)
                    val_loss = torch.mean(torch.tensor(loss_list))
                    if val_loss.item() < init_loss:
                        init_loss = val_loss.item()
                        rec_mkdir(save_path)  # 保证该路径下文件夹存在
                        torch.save(model.state_dict(), save_path)
                        print("保存模型:{}....val_loss:{:.4e}".format(save_path, init_loss))
                    if val_loss.item() < 1e-7:
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
        torch.cuda.empty_cache()  # 清空缓存
        bar = tqdm(data_loader)
        for idx, data in enumerate(bar):
            optimizer.zero_grad()
            data = data.to(config.device)
            teacher_ouput = teacher(data, None)
            stu_output = stu(data, None)
            loss = F.mse_loss(stu_output, teacher_ouput)
            loss.backward()
            nn.utils.clip_grad_norm_(stu.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()
            bar.set_description(info + "\tepoch:{}\tidx:{}\tloss:{:.4e}".format(i + 1, idx, loss.item()))
        # 模型验证
        if (idx + 1) % 10 == 0:
            with torch.no_grad():
                loss_list = list()
                for idx, batch_x in enumerate(comm_val_dataloader):
                    batch_x = batch_x.to(config.device)
                    output = stu(batch_x, None)
                    batch_x = batch_x - 0.5
                    output = output - 0.5
                    cur_loss = F.mse_loss(output, batch_x).item()
                    loss_list.append(cur_loss)
                val_loss = torch.mean(torch.tensor(loss_list))
                if val_loss.item() < init_loss:
                    init_loss = val_loss.item()
                    rec_mkdir(save_path)  # 保证该路径下文件夹存在
                    torch.save(stu.state_dict(), save_path)
                    print("保存模型:{}....val_loss:{:.4e}".format(save_path, init_loss))
                if val_loss.item() < 1e-7:
                    return
