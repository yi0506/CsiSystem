"""模型训练"""
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import torch
import multiprocessing
import torch.nn as nn
import os

from libs.csi_dataset import data_load
from libs import config
from libs.model import Seq2Seq, Encoder, Decoder, Noise


def train_model_merged(epoch, snr, ratio, velocity):
    """在一种信噪比下进行模型训练，模型为一个整体"""
    my_model = Seq2Seq(snr=snr, ratio=ratio).to(config.device)
    my_model.train()
    optimizer = Adam(my_model.parameters())
    data_loader = data_load(True, velocity)
    init_loss = 1
    for i in range(epoch):
        bar = tqdm(data_loader)
        for idx, data in enumerate(bar):
            optimizer.zero_grad()  # 梯度置为零
            data = data.to(config.device)  # 转到GPU训练
            output = my_model(data)
            similarity = torch.cosine_similarity(output, data, dim=-1).mean()  # 当前一个batch的相似度
            loss = F.mse_loss(output, data)  # 计算损失
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(my_model.parameters(), config.clip)  # 进行梯度裁剪
            optimizer.step()  # 梯度更新
            bar.set_description("v:{}\tratio:{}\tSNR:{}dB\tepoch:{}\tindex:{}\tloss:{:}\tsimilarity:{:.3f}"
                                .format(velocity, ratio, snr, i + 1, idx, loss.item(), similarity.item()))
            if loss.item() < init_loss:
                init_loss = loss.item()
                model_save_path = "./model/{}km/ratio_{}/{}dB.model".format(velocity, ratio, snr)
                rec_mkdir(model_save_path)  # 保证该路径下文件夹存在
                torch.save(my_model.state_dict(), model_save_path)
            if loss.item() < 1e-6:
                return


def train_model_separated(epoch, snr, ratio, velocity):
    """在一种信噪比下进行模型训练，每个模块分开保存"""
    noise = Noise(snr=snr, ratio=ratio).to(config.device).train()
    encoder = Encoder(ratio).to(config.device).train()
    decoder = Decoder(ratio).to(config.device).train()
    optim_encoder = Adam(encoder.parameters())
    optim_noise = Adam(noise.parameters())
    optim_decoder = Adam(decoder.parameters())
    data_loader = data_load(True, velocity)
    init_loss = 1
    for i in range(epoch):
        bar = tqdm(data_loader)
        for idx, data in enumerate(bar):
            optim_encoder.zero_grad()  # 梯度置为零
            optim_noise.zero_grad()
            optim_decoder.zero_grad()
            data = data.to(config.device)  # 转到GPU训练
            out = encoder(data)
            out = noise(out)
            output = decoder(out)
            similarity = torch.cosine_similarity(output, data, dim=-1).mean()  # 当前一个batch的相似度
            loss = F.mse_loss(output, data)  # 计算损失
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(encoder.parameters(), config.clip)  # 进行梯度裁剪
            nn.utils.clip_grad_norm_(noise.parameters(), config.clip)
            nn.utils.clip_grad_norm_(decoder.parameters(), config.clip)
            optim_encoder.step()  # 梯度更新
            optim_noise.step()
            optim_decoder.step()
            bar.set_description("v:{}\tratio:{}\tSNR:{}dB\tepoch:{}\tindex:{}\tloss:{:}\tsimilarity:{:.3f}"
                                .format(velocity, ratio, snr, i + 1, idx, loss.item(), similarity.item()))
            if loss.item() < init_loss:
                init_loss = loss.item()
                encoder_save_path = "./model/{}km/ratio_{}/separate/{}dB.encoder".format(velocity, ratio, snr)
                noise_save_path = "./model/{}km/ratio_{}/separate/{}dB.noise".format(velocity, ratio, snr)
                decoder_save_path = "./model/{}km/ratio_{}/separate/{}dB.decoder".format(velocity, ratio, snr)
                rec_mkdir(encoder_save_path)
                rec_mkdir(decoder_save_path)
                rec_mkdir(noise_save_path)
                torch.save(encoder.state_dict(), encoder_save_path)
                torch.save(noise.state_dict(), noise_save_path)
                torch.save(decoder.state_dict(), decoder_save_path)
            if loss.item() < 1e-5:
                return


def concurrent_train(epoch, model_list, ratio, velocity, multi=True):
    """使用多进程训练模型"""
    torch.cuda.empty_cache()  # 清空缓存
    if multi:
        pool = multiprocessing.Pool(6)
        for snr in model_list:
            pool.apply_async(func=train_model_merged, args=(epoch, snr, ratio, velocity))
        pool.close()
        pool.join()
    else:
        for snr in model_list:
            train_model_merged(epoch, snr, ratio, velocity)

def rec_mkdir(file_path):
    """递归创建文件夹，如果文件夹不存在，就创建，存在就跳过，继续下一层"""
    dir_list = file_path.split('/')[:-1]
    cur_dir = dir_list[0] + "/"
    for dir in dir_list[1:]:
        cur_dir += dir
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        cur_dir += "/"


if __name__ == '__main__':
    dir_path = "../model/{}km/ratio_{}/epoch/old_csi/epoch/xxx.db".format(200, 16, 5)
    rec_mkdir(dir_path)

