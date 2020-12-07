# -*- coding: UTF-8 -*-
"""原始csi_net网络模型的构建、训练、测试并保存结果"""
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import time
import numpy as np
import torch

from libs.train import rec_mkdir
from libs import config
from libs.csi_dataset import data_load


class CsiNet(nn.Module):
    """压缩端与解压缩端合并封装"""

    def __init__(self, **kwargs):
        """初始化模型与必要参数"""
        super(CsiNet, self).__init__()
        self.ratio = kwargs.get("ratio", config.old_csi_net_compress_ratio)  # 压缩率
        self.encoder = Encoder(self.ratio)
        self.decoder = Decoder(self.ratio)
        self.add_noise = kwargs.get("add_noise", False)  # 是否加入噪声
        self.snr = kwargs.get("snr", None)  # 信噪比

    def forward(self, input_):
        """前向传播过程"""
        if self.add_noise is False:
            encoder_output = self.encoder(input_)
            decoder_output = self.decoder(encoder_output)
            return decoder_output
        else:
            encoder_output = self.encoder(input_)
            output_add_noise = self.wgn(encoder_output)  # 信号加入噪声
            decoder_output = self.decoder(output_add_noise)
            return decoder_output

    def wgn(self, x):
        """
        对信号加入高斯白噪声,
        噪音强度为：10 ** (snr / 10)，其中snr为对应的信噪比
        注：信号的幅度的平方==信号的功率，因此要开平方根
        :param x: 信号
        :return:
        """
        x_power = (torch.sum(x ** 2) / x.numel())  # 信号的功率
        noise_power = (x_power / torch.tensor(10 ** (self.snr / 10)))  # 噪声的功率
        gaussian = torch.normal(0, torch.sqrt(noise_power).item(), (x.size()[0], x.size()[1]))  # 产生对应信噪比的高斯白噪声
        return x + gaussian.to(config.device)


class Encoder(nn.Module):
    """压缩"""
    channel_num = config.old_csi_channel_num  # 信道矩阵通道数
    data_length = config.old_csi_data_length  # 信道矩阵元素数量
    slope = config.old_csi_slope  # LeakRule的负斜率

    def __init__(self, ratio):
        super(Encoder, self).__init__()
        self.ratio = ratio  # 压缩率
        self.csi_conv1 = nn.Sequential(
                                nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm1d(self.channel_num),
                                nn.LeakyReLU(self.slope)
                            )
        self.fc = nn.Linear(self.data_length, int(self.data_length / self.ratio))

    def forward(self, input_):
        """
        由于输入数据形状为[batch_size, 32*32]，
        因此需要先reshape操作：[batch_size, 32*32] -----> [batch_size, 32, 32]，
        后续与csi—net一致，注释表示输出数据的形状

        压缩：reshape ---> 卷积 ---> 全连接压缩

        :param input_: [batch_size, 32*32]
        :return: [batch_size, 32*32/8]
        """
        # reshape
        out = input_.view(-1, self.channel_num, self.channel_num)  # [batch_size, 32, 32]
        # 卷积
        out = self.csi_conv1(out)  # [batch_size, 32, 32]
        # 全连接
        out = out.view(-1, self.data_length)  # [batch_size, 32*32]
        output = self.fc(out)  # [batch_size, 32*32/ratio]
        return output


class Decoder(nn.Module):
    """解压缩"""
    data_length = config.old_csi_data_length  # 信道矩阵元素数量
    channel_num = config.old_csi_channel_num  # 压缩率
    slope = config.old_csi_slope  # LeakRule的负斜率

    def __init__(self, ratio):
        super(Decoder, self).__init__()
        self.ratio = ratio  # 信道矩阵通道数
        self.fc = nn.Linear(int(self.data_length / self.ratio), self.data_length)
        self.leaky_relu = nn.LeakyReLU(self.slope)
        self.refine_net_1 = nn.Sequential(
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm1d(self.channel_num),
                                    nn.LeakyReLU(0.3),
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm1d(self.channel_num),
                                    nn.LeakyReLU(0.3),
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm1d(self.channel_num)
                                )
        self.refine_net_2 = nn.Sequential(
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm1d(self.channel_num),
                                    nn.LeakyReLU(0.3),
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm1d(self.channel_num),
                                    nn.LeakyReLU(0.3),
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm1d(self.channel_num)
                                )
        self.output_conv = nn.Sequential(
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm1d(self.channel_num),
                                    nn.Sigmoid()
                                )

    def forward(self, encoder_output):
        """
        由于训练数据形状为[batch_size, 32*32]，
        因此最后需要reshape操作：[batch_size, 32, 32] -----> [batch_size, 32*32]，变为原来的形状

        数据解压缩：
        全连接 ----> reshape ----> 残差（refine_net） ----> 残差（refine_net） -----> 卷积  -----> reshape

        :param encoder_output: [batch_size, 32*32/ratio]
        :return: [batch_size, 32*32]
        """

        # 全连接  [batch_size, 32*32/ratio] -----> [batch_size, 32*32]
        out = self.fc(encoder_output)

        # reshape  [batch_size, 32*32] -----> [batch_size, 32, 32]
        out = out.view(-1, self.channel_num, self.channel_num)

        # refine_net  [batch_size, 32, 32] -----> [batch_size, 32, 32]
        out = self.res_unit(self.refine_net_1, out)
        out = self.leaky_relu(out)
        out = self.res_unit(self.refine_net_2, out)

        # 卷积  [batch_size, 32, 32] -----> [batch_size, 32, 32]
        out = self.output_conv(out)

        # reshape  [batch_size, 32, 32] -----> [batch_size, 32*32]
        output = out.view(-1, self.data_length)

        return output

    @staticmethod
    def res_unit(func, input_):
        """残差结构"""
        out = func(input_)
        output = input_ + out
        return output


class Trainer(object):
    """csi_net训练器"""
    device = config.device

    def __init__(self, epoch, **kwargs):
        """初始化模型、数据集、优化器、必要参数"""
        self.ratio = kwargs.get("ratio", config.old_csi_net_compress_ratio)
        self.velocity = kwargs.get('velocity', config.velocity)
        self.model_path = kwargs.get("model_path", "./model/{}km/ratio_{}/old_csi/old_csi_{}.pt".format(self.velocity, self.ratio, self.ratio))
        self.epoch = epoch
        self.data_loader = data_load(True, self.velocity)
        self.model = CsiNet(ratio=self.ratio).to(self.device).train()
        self.optimizer = Adam(self.model.parameters())

    def run(self):
        """训练模型"""
        rec_mkdir(self.model_path)
        for i in range(self.epoch):
            for idx, data in enumerate(self.data_loader):
                self.optimizer.zero_grad()  # 梯度置为零
                data = data.to(self.device)  # 转到GPU训练
                output = self.model(data)
                similarity = torch.cosine_similarity(output, data, dim=-1).mean()  # 一个batch的相似度
                loss = F.mse_loss(output, data)  # 一个batch的损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 梯度更新
                self.save_model(idx, loss.item())  # 保存模型

                # 打印进度条
                total_num = len(self.data_loader)
                cur_num = idx + 1
                rest_process_bar = (len(self.data_loader) - idx) * " "
                cur_process_bar = idx * "#"
                process_bar = cur_process_bar + rest_process_bar
                process_bar = "|" + process_bar + "|"
                print("\rv:{}\tratio:{}\tepoch:{}\tidx:{}\t{} {}|{}\tloss:{:}\tsimilarity:{:.3f}".format(self.velocity, self.ratio, i + 1, idx, process_bar,cur_num, total_num, loss.item(),similarity.item()), end="")
            print()

    def save_model(self, idx, loss):
        """保存模型"""
        # 10个batch存一次
        if (idx + 1) % 10 == 0:
            torch.save(self.model.state_dict(), self.model_path)
        if loss < 1e-5:
            return

    def __call__(self, *args, **kwargs):
        return self.run()


class Tester(object):
    """csi_net测试器"""
    device = config.device
    batch_size = config.test_batch_size

    def __init__(self, **kwargs):
        """加载模型、数据集、必要参数"""
        self.ratio = kwargs.get("ratio", config.old_csi_net_compress_ratio)  # 压缩率
        add_noise = kwargs.get("add_noise", False)  # 是否加入噪声
        self.snr = kwargs.get("snr", None)  # 信噪比
        self.velocity = kwargs.get("velocity", config.velocity)  # 速度
        self.model_path = kwargs.get("model_path", "./model/{}km/ratio_{}/old_csi/old_csi_{}.pt".format(self.velocity, self.ratio, self.ratio))
        self.model = CsiNet(ratio=self.ratio, add_noise=add_noise, snr=self.snr).to(self.device).eval()
        self.model.load_state_dict(torch.load(self.model_path))
        self.data_loader = tqdm(data_load(False, self.velocity))

    def run(self):
        """测试模型"""
        loss_list = list()
        similarity_list = list()
        time_list = list()
        for idx, input_ in enumerate(self.data_loader):
            with torch.no_grad():
                start_time = time.time()
                input_ = input_.to(self.device)
                output = self.model(input_)
                stop_time = time.time()
                cur_similarity = torch.cosine_similarity(output, input_, dim=-1).mean().cpu().item()
                cur_loss = F.mse_loss(output, input_).cpu().item()  # 一个batch的损失
                loss_list.append(cur_loss / self.batch_size)  # 处理一个信号的损失
                similarity_list.append(cur_similarity)  # 处理一个信号的相似度
                time_list.append((stop_time - start_time) / self.batch_size)  # 处理一个信号的时间
        return self.save_result(loss_list, similarity_list, time_list)

    def save_result(self, loss, similarity, time) -> dict:
        """保存测试结果"""
        # 计算平均相似度与损失
        avg_loss = np.mean(loss)
        avg_similarity = np.mean(similarity)
        avg_time = np.mean(time)
        result = {"ratio": self.ratio, "相似度": avg_similarity, "NMSE": avg_loss, "time": avg_time}
        return result


if __name__ == '__main__':
    model = CsiNet().to(config.device)
    summary(model, (1024,))
