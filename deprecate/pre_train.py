# -*- coding: UTF-8 -*-
"""
@author:wsy123
@file:pre_train.py
@time:2020/07/19

网络分模块训练
"""

from tqdm import tqdm
from torch.optim import Adam
from lib.csi_dataset import data_load
from lib import config
import torch
import multiprocessing
from lib.model import Noise, Encoder, Decoder
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def pre_train_noise(epoch, snr, ratio):
    """对一种信噪比进行训练noise网络"""
    noise = Noise(snr=snr, ratio=ratio).to(config.device)
    noise.train()
    optimizer = Adam(noise.parameters())
    data_loader = data_load()
    for i in range(epoch):
        bar = tqdm(data_loader)
        for idx, _ in enumerate(bar):
            optimizer.zero_grad()  # 梯度置为零
            np.random.seed(1)
            data = torch.tensor(np.random.randn(config.train_batch_size, int(config.data_length / ratio)),
                                dtype=torch.float).to(config.device)
            output = noise(data)  # [batch_size, 32*32]
            similarity = torch.cosine_similarity(output, data, dim=-1).mean()  # 当前一个batch的相似度
            mse_loss = F.mse_loss(output, data)
            mse_loss.backward()
            nn.utils.clip_grad_norm_(noise.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()  # 梯度更新
            bar.set_description("noise--ratio:{}\tnoise_SNR:{}dB\tepoch:{}\tindex:{}\tloss:{:}\tsimilarity:{:.3f}"
                                .format(ratio, snr, i + 1, idx + 1, mse_loss.item(), similarity.item()))
            # 10个batch存一次
            if (idx + 1) % 10 == 0:
                model_save_path = "./model/ratio_{}/pre_train/noise/{}dB.noise".format(ratio, snr)
                torch.save(noise.state_dict(), model_save_path)
            if mse_loss.item() < 1e-7:
                return
        # 50个epoch存一次
        if (i + 1) % 50 == 0:
            save_path_ = "./model/ratio_{}/pre_train/noise/epoch/epoch_{}_{}dB.noise".format(ratio, i + 1, snr)
            torch.save(noise.state_dict(), save_path_)


def pre_train_seq2seq(epoch, ratio):
    """对一种信噪比进行训练seq2seq网络，不加入噪声网络"""
    encoder = Encoder(ratio).to(config.device).train()
    decoder = Decoder(ratio).to(config.device).train()
    optimizer1 = Adam(encoder.parameters())
    optimizer2 = Adam(decoder.parameters())
    data_loader = data_load()
    for i in range(epoch):
        bar = tqdm(data_loader, desc="seq2seq")
        for idx, data in enumerate(bar):
            optimizer1.zero_grad()  # 梯度置为零
            optimizer2.zero_grad()  # 梯度置为零
            data = data.to(config.device)  # 转到GPU训练
            encoder_output = encoder(data)
            decoder_output = decoder(encoder_output)
            similarity = torch.cosine_similarity(decoder_output, data, dim=-1).mean()  # 当前一个batch的相似度
            mse_loss = F.mse_loss(decoder_output, data)  # 计算损失
            mse_loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(encoder.parameters(), config.clip)  # 进行梯度裁剪
            nn.utils.clip_grad_norm_(decoder.parameters(), config.clip)  # 进行梯度裁剪
            optimizer1.step()  # 梯度更新
            optimizer2.step()  # 梯度更新
            bar.set_description("seq2seq_\tratio:{}\tepoch:{}\tindex:{}\tloss:{:}\tsimilarity:{:.3f}"
                                .format(ratio, i + 1, idx, mse_loss.item(), similarity.item()))
            # 10个batch存一次
            if (idx + 1) % 10 == 0:
                encoder_model_path = "./model/ratio_{}/pre_train/seq2seq/encoder.pkl".format(ratio)
                decoder_model_path = "./model/ratio_{}/pre_train/seq2seq/decoder.pkl".format(ratio)
                torch.save(encoder.state_dict(), encoder_model_path)
                torch.save(decoder.state_dict(), decoder_model_path)
            if mse_loss.item() < 1e-6:
                return
        # 50个epoch存一次
        if (i + 1) % 50 == 0:
            encoder_model_epoch_path = "./model/ratio_{}/pre_train/seq2seq/epoch/epoch_{}_encoder.pkl".format(
                ratio, i + 1)
            decoder_model_epoch_path = "./model/ratio_{}/pre_train/seq2seq/epoch/epoch_{}_decoder.pkl".format(
                ratio, i + 1)
            torch.save(encoder.state_dict(), encoder_model_epoch_path)
            torch.save(decoder.state_dict(), decoder_model_epoch_path)


def overfit_one_batch_train_seq2seq(epoch, ratio):
    """
    将每个batch的训练数据设置为相同数据，进行过拟合训练
    通过过拟合训练，测试网络的学习能力，判断网络是否能够学习所需特征，恢复出原始信号
    """
    encoder = Encoder(ratio).to(config.device).train()
    decoder = Decoder(ratio).to(config.device).train()
    optimizer1 = Adam(encoder.parameters())
    optimizer2 = Adam(decoder.parameters())
    for i in tqdm(range(epoch)):
        data_loader = data_load()
        data = next(iter(data_loader))
        optimizer1.zero_grad()  # 梯度置为零
        optimizer2.zero_grad()  # 梯度置为零
        data = data.to(config.device)  # 转到GPU训练
        encoder_output = encoder(data)
        decoder_output = decoder(encoder_output)
        print("原始数据:\n", data)
        print("*" * 40)
        print("输出：\n", decoder_output)
        similarity = torch.cosine_similarity(decoder_output, data, dim=-1).mean()  # 当前一个batch的相似度
        mse_loss = F.mse_loss(decoder_output, data)  # 计算损失
        mse_loss.backward()  # 反向传播
        nn.utils.clip_grad_norm_(encoder.parameters(), config.clip)  # 进行梯度裁剪
        nn.utils.clip_grad_norm_(decoder.parameters(), config.clip)  # 进行梯度裁剪
        optimizer1.step()  # 梯度更新
        optimizer2.step()  # 梯度更新
        print("相似度：", similarity.item(), "\t", "loss:", mse_loss.item())

        # 模型的保存
        if (i + 1) % 50 == 0:
            encoder_model_epoch_path = "./model/ratio_{}/pre_train/seq2seq/epoch/epoch_{}_encoder.pkl".format(ratio, i + 1)
            decoder_model_epoch_path = "./model/ratio_{}/pre_train/seq2seq/epoch/epoch_{}_decoder.pkl".format(ratio, i + 1)
            torch.save(encoder.state_dict(), encoder_model_epoch_path)
            torch.save(decoder.state_dict(), decoder_model_epoch_path)
        if mse_loss.item() < 1e-6:
            break


def pre_train_noise_concurrent(epoch, ratio, model_list, multi=False):
    """
    不同信噪比下训练noise模型
    :param model_list: 选择的不同信噪比模型
    :param ratio: 压缩率
    :param epoch: 训练次数
    :param multi:是否使用多进程
    """
    torch.cuda.empty_cache()  # 清空缓存
    if multi:
        pool = multiprocessing.Pool(7)
        for snr in model_list:
            pool.apply_async(func=pre_train_noise, args=(epoch, snr, ratio))
        pool.close()
        pool.join()
    else:
        for snr in model_list:
            pre_train_noise(epoch, snr, ratio)


if __name__ == '__main__':
    pass
