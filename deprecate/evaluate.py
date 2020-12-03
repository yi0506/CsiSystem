# -*- coding: UTF-8 -*-
"""测试每个网络模块的效果"""
import torch
from libs.model import Noise, Encoder, Decoder
from torch.nn.functional import cosine_similarity, mse_loss
from libs import config
from libs.csi_dataset import data_load
from tqdm import tqdm
import numpy as np
import time
import pickle
import os


def pre_test_seq2seq(ratio):
    """评估seq2seq模型，不加入去噪声网络"""

    # 实例化模型
    encoder = Encoder(ratio).to(config.device).eval()
    decoder = Decoder().to(config.device).eval()
    # 加载模型
    encoder_model_path = "./model/ratio_{}/pre_train/seq2seq/encoder.pkl".format(ratio)
    decoder_model_path = "./model/ratio_{}/pre_train/seq2seq/decoder.pkl".format(ratio)
    encoder.load_state_dict(torch.load(encoder_model_path))
    decoder.load_state_dict(torch.load(decoder_model_path))
    loss_list = list()
    similarity_list = list()
    data_loader = data_load()
    for idx, input_ in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            input_ = input_.to(config.device)
            encoder_output = encoder(input_)
            decoder_output = decoder(encoder_output)
            # 当前一个batch的相似度
            cur_similarity = cosine_similarity(decoder_output, input_, dim=-1).mean().cpu().item()
            # 当前一个batch的损失
            cur_loss = mse_loss(decoder_output, input_).cpu().item()
            loss_list.append(cur_loss / config.train_batch_size)
            similarity_list.append(cur_similarity)

    # 计算平均相似度与损失
    avg_loss = np.mean(loss_list)
    avg_similarity = np.mean(similarity_list)
    print("seq2seq\tloss:{:.3f}\tsimilarity:{:.3f}".format(avg_loss, avg_similarity))


def test_noise_with_different_snr(model_list, ratio=config.net_compress_ratio, velocity=config.velocity):
    result_dict = dict()
    for snr_model in model_list:
        model_path = "./model/{}km/ratio_{}/pre_train/noise/{}dB.noise".format(velocity, ratio, snr_model)
        if not os.path.exists(model_path):
            print("模型不存在")
            return
        for snr in config.SNRs:
            result = pre_test_noise(snr, ratio, model_path)
            result_dict["{}dB".format(snr)] = result
        file_path = "./test_result/{}km/noise/noise_{}_{}dB.pkl".format(velocity, ratio, snr_model)
        pickle.dump(result_dict, open(file_path, "wb"))


def pre_test_noise(snr, ratio, model_path, velocity=config.velocity) -> dict:
    """评估去噪网络模型"""

    # 实例化模型
    noise = Noise(snr, ratio).to(config.device)
    noise.eval()
    # 加载模型
    noise.load_state_dict(torch.load(model_path))
    loss_list = list()
    similarity_list = list()
    time_list = list()
    data_loader = data_load()
    for idx, _ in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            input_ = torch.tensor(
                np.random.randn(config.train_batch_size, int(config.data_length / ratio)),
                dtype=torch.float).to(config.device)
            start_time = time.time()
            output = noise(input_)
            stop_time = time.time()
            # 当前一个batch的相似度
            cur_similarity = cosine_similarity(output, input_, dim=-1).mean().cpu().item()
            # 当前一个batch的损失
            cur_loss = mse_loss(output, input_).cpu()
            loss_list.append(cur_loss / config.train_batch_size)
            similarity_list.append(cur_similarity)
            time_list.append((stop_time - start_time) / config.train_batch_size)

    # 计算平均相似度与损失
    avg_loss = np.mean(loss_list)
    avg_similarity = np.mean(similarity_list)
    avg_time = np.mean(time_list)
    print("noise\tv:{}\tSNR:{}\tloss:{:.3f}\tsimilarity:{:.3f}\ttime:{:.4f}".format(velocity, snr, avg_loss, avg_similarity, avg_time))
    result = {"NMSE": avg_loss, "相似度": avg_similarity, "time": avg_time}
    return result


def test_one_batch_under_noise_model(snr, ratio):
    """测试一个batch的信号在去噪网络模型下的效果"""
    # 实例化模型
    noise = Noise(snr, ratio).to(config.device)
    noise.eval()
    noise_model_path = "./model/ratio_{}/pre_train/noise/{}dB.noise".format(ratio, snr)
    # 加载模型
    noise.load_state_dict(torch.load(noise_model_path))
    with torch.no_grad():
        data = torch.randn(config.train_batch_size, int(config.data_length / ratio)).to(config.device)
        output = noise(data)
        print(data)
        print(output)
        # 当前一个batch的相似度
        cur_similarity = cosine_similarity(output, data, dim=-1).mean().cpu()
        # 当前一个batch的损失
        cur_loss = mse_loss(output, data).cpu()
        print("相似度\t", cur_similarity.item(), "\t损失\t", cur_loss.item())


def overfit_one_batch_test_seq2seq(ratio):
    """评估过拟合模型的效果"""
    # 实例化模型
    encoder = Encoder(ratio).to(config.device).eval()
    decoder = Decoder().to(config.device).eval()
    # 加载模型
    encoder_model_path = "./model/ratio_{}/pre_train/seq2seq/encoder.pkl".format(ratio)
    decoder_model_path = "./model/ratio_{}/pre_train/seq2seq/decoder.pkl".format(ratio)
    encoder.load_state_dict(torch.load(encoder_model_path))
    decoder.load_state_dict(torch.load(decoder_model_path))
    data_loader = data_load()
    input_ = next(iter(data_loader)).to(config.device)
    print("\n", input_)
    with torch.no_grad():
        encoder_output = encoder(input_)
        decoder_output = decoder(encoder_output)
        # 当前一个batch的相似度
        cur_similarity = cosine_similarity(decoder_output, input_, dim=-1).mean().cpu().item()
        # 当前一个batch的损失
        cur_loss = mse_loss(decoder_output, input_).cpu().item()

    print("one_batch\tloss:{:.3f}\tsimilarity:{:.3f}".format(cur_loss, cur_similarity))
    print("\n", decoder_output)


if __name__ == "__main__":
    pass
