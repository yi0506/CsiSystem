"""模型测试"""
import torch
from torch.nn.functional import cosine_similarity, mse_loss
from libs.train import rec_mkdir
from tqdm import tqdm
import numpy as np
import pickle
from multiprocessing import Pool
import time

from libs import config
from libs.csi_dataset import data_load
from libs.model import Seq2Seq, Encoder, Decoder, Noise


def test_model_separated(snr_model, ratio, velocity):
    """snr_model：选择某种信噪比模型"""
    result_dict = dict()
    for snr in config.SNRs:
        noise = Noise(snr=snr, ratio=ratio).to(config.device).eval()
        encoder = Encoder(ratio).to(config.device).eval()
        decoder = Decoder().to(config.device).eval()
        encoder_model_path = "./model/{}km/ratio_{}/separate/{}dB.encoder".format(velocity, ratio, snr_model)
        noise_model_path = "./model/{}km/ratio_{}/separate/{}dB.noise".format(velocity, ratio, snr_model)
        decoder_model_path = "./model/{}km/ratio_{}/separate/{}dB.decoder".format(velocity, ratio, snr_model)
        encoder.load_state_dict(torch.load(encoder_model_path))
        noise.load_state_dict(torch.load(noise_model_path))
        decoder.load_state_dict(torch.load(decoder_model_path))
        loss_list = list()
        similarity_list = list()
        time_list = list()
        data_loader = data_load(False, velocity)
        for idx, input_ in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                start_time = time.time()
                input_ = input_.to(config.device)
                out = encoder(input_)
                out = noise(out)
                output = decoder(out)
                stop_time = time.time()
                cur_similarity = cosine_similarity(output, input_, dim=-1).mean().cpu().item()  # 当前一个信号的平均的相似度
                cur_loss = mse_loss(output, input_).cpu().item()  # 当前一个batch的损失
                loss_list.append(cur_loss / config.test_batch_size)
                similarity_list.append(cur_similarity)
                time_list.append((stop_time - start_time) / config.test_batch_size)

        # 计算平均相似度与损失
        avg_loss = np.mean(loss_list)
        avg_similarity = np.mean(similarity_list)
        avg_time = np.mean(time_list)
        print("v:{}\tratio:{}\tSNR:{}dB\tloss:{:.3f}\tsimilarity:{:.3f}\ttime:{:.4f}".format(velocity, ratio, snr, avg_loss, avg_similarity, avg_time))
        result = {"相似度": avg_similarity, "NMSE": avg_loss, "time": avg_time}
        result_dict["{}dB".format(snr)] = result
    file_path = "./test_result/{}km/csi_net/separate/csi_net_{}.pkl".format(velocity, ratio)
    rec_mkdir(file_path)
    pickle.dump(result_dict, open(file_path, "wb"))


def test(snr_model, ratio, velocity):
    """评估模型，snr_model：选择某种模型"""
    result_dict = dict()
    for snr in config.SNRs:
        # 实例化模型
        seq2seq = Seq2Seq(snr, ratio).to(config.device).eval()
        # 加载模型
        seq2seq_model_path = "./model/{}km/ratio_{}/{}dB.model".format(velocity, ratio, snr_model)
        seq2seq.load_state_dict(torch.load(seq2seq_model_path), strict=False)
        loss_list = list()
        similarity_list = list()
        time_list = list()
        capacity_list = list()
        data_loader = data_load(False, velocity)
        for idx, input_ in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                start_time = time.time()
                input_ = input_.to(config.device)
                output = seq2seq(input_)
                stop_time = time.time()
                cur_similarity = cosine_similarity(output, input_, dim=-1).mean().cpu().item()
                cur_loss = mse_loss(output, input_).cpu().item()
                cur_capacity = torch.log2(torch.sum(1 + torch.linalg.svd(output)[1] * snr / config.Nt)).item() * config.net_capacity_ratio  # 信道容量:SVD分解
                capacity_list.append(cur_capacity / config.test_batch_size)
                loss_list.append(cur_loss / config.test_batch_size)
                similarity_list.append(cur_similarity)
                time_list.append((stop_time - start_time) / config.test_batch_size)

        # 计算平均相似度与损失
        avg_loss = np.mean(loss_list)
        avg_similarity = np.mean(similarity_list)
        avg_time = np.mean(time_list)
        avg_capacity = np.mean(capacity_list)
        print("v:{}\tratio:{}\tSNR:{}dB\tloss:{:.3f}\tsimilarity:{:.3f}\ttime:{:.4f}\tcapacity:{:.4f}".format(velocity, ratio, snr, avg_loss, avg_similarity, avg_time, avg_capacity))
        result = {"相似度": avg_similarity, "NMSE": avg_loss, "time": avg_time, "Capacity": avg_capacity}
        result_dict["{}dB".format(snr)] = result
    file_path = "./test_result/{}km/csi_net/csi_net_{}_{}dB.pkl".format(velocity, ratio, snr_model)
    rec_mkdir(file_path)
    pickle.dump(result_dict, open(file_path, "wb"))


def concurrent_test(model_list, ratio, velocity, multi=True):
    """
    评估模型
    :param velocity: 选用哪种速度下的数据集
    :param model_list: 不同信噪比模型
    :param ratio: 压缩率
    :param multi: 是否使用多进程
    """
    torch.cuda.empty_cache()  # 清空缓存
    if multi:
        pool = Pool(6)
        for snr in model_list:
            pool.apply_async(func=test, args=(snr, ratio, velocity))
        pool.close()
        pool.join()
    else:
        for snr in model_list:
            test(snr, ratio, velocity)


if __name__ == "__main__":
    pass
