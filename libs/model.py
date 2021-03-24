"""csi网络模型"""
import torch
import torch.nn as nn
from torchsummary import summary

from libs import config


class BaseModel(nn.Module):
    """模型类基类"""
    data_length = config.data_length
    channel_num = config.channel_num
    conv_group = config.conv_group
    channel_multiple = config.channel_multiple

    @staticmethod
    def res_unit(func, input_):
        """用过残差网络提取特征"""
        out = func(input_)
        output = out + input_  # 加入残差结构
        return output

    @staticmethod
    def normalize(input_):
        """标准化处理"""
        mean = torch.mean(input_, dim=-1, keepdim=True)
        std = torch.std(input_, dim=-1, keepdim=True)
        output = (input_ - mean) / std
        return output


class Seq2Seq(BaseModel):
    """训练模型三个部分合并封装"""
    def __init__(self, snr, ratio):
        super(Seq2Seq, self).__init__()
        self.ratio = ratio
        self.encoder = Encoder(ratio)
        self.noise = Noise(snr, ratio)
        self.decoder = Decoder()

    def forward(self, input_):
        encoder_output = self.encoder(input_)
        add_noise = self.noise(encoder_output)
        decoder_output = self.decoder(add_noise)
        return decoder_output


class Encoder(BaseModel):
    """MS压缩"""

    def __init__(self, ratio):
        super(Encoder, self).__init__()
        self.ratio = ratio
        self.group_conv1 = nn.Sequential(
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num * self.channel_multiple,
                                              groups=self.conv_group, kernel_size=3, stride=1, padding=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num * self.channel_multiple),
                                    nn.Conv1d(in_channels=self.channel_num * self.channel_multiple, out_channels=self.channel_num,
                                              groups=self.conv_group, kernel_size=3, stride=1, padding=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num),
                                )
        self.group_conv2 = nn.Sequential(
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num * self.channel_multiple,
                                              groups=self.conv_group, kernel_size=3, stride=1, padding=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num * self.channel_multiple),
                                    nn.Conv1d(in_channels=self.channel_num * self.channel_multiple, out_channels=self.channel_num,
                                              groups=self.conv_group, kernel_size=3, stride=1, padding=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num),
                                )
        self.fc_out = nn.Sequential(
                                nn.Linear(self.data_length, int(self.data_length / self.ratio)),
                                nn.ELU(True),
                                nn.BatchNorm1d(int(self.data_length / self.ratio)),
                        )

    def forward(self, input_):
        """
        压缩：标准化 ----> 全连接 ----> 残差（分组卷积） ---> 残差（分组卷积） ---> 全连接压缩
        :param input_: [batch_size, 32*32]
        :return: [batch_size, 32*32/ratio]
        """
        # 标准化
        out = self.normalize(input_)  # [batch_size, 32*32]
        out = out.view(-1, self.channel_num, self.channel_num)  # [batch_size, 32, 32]
        # 分组卷积
        out = self.res_unit(self.group_conv1, out)  # [batch_size, 32, 32]
        out = self.res_unit(self.group_conv2, out)  # [batch_size, 32, 32]
        out = out.view(-1, self.channel_num * self.channel_num)  # [batch_size, 32*32]
        # 全连接
        output = self.fc_out(out)  # [batch_size, 32*32/ratio]
        return output


class Decoder(BaseModel):
    """解压缩"""

    def __init__(self):
        super(Decoder, self).__init__()
        self.deep_conv = nn.Sequential(
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num,
                                              groups=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num,
                                              groups=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num),
                                )
        self.group_conv1 = nn.Sequential(
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num * self.channel_multiple,
                                              groups=self.conv_group, kernel_size=3, stride=1, padding=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num * self.channel_multiple),
                                    nn.Conv1d(in_channels=self.channel_num * self.channel_multiple, out_channels=self.channel_num,
                                              groups=self.conv_group, kernel_size=3, stride=1, padding=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num),
                                )
        self.deep_separate = nn.Sequential(
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num,
                                              groups=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num * self.channel_multiple,
                                              kernel_size=1, stride=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num * self.channel_multiple),
                                    nn.Conv1d(in_channels=self.channel_num * self.channel_multiple, out_channels=self.channel_num,
                                              groups=self.conv_group, kernel_size=3, stride=1, padding=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num),
                                )
        self.deep_separate_2 = nn.Sequential(
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num,
                                              groups=self.channel_num, kernel_size=3, stride=1, padding=1),
                                    nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num * self.channel_multiple,
                                              kernel_size=1, stride=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num * self.channel_multiple),
                                    nn.Conv1d(in_channels=self.channel_num * self.channel_multiple, out_channels=self.channel_num,
                                              groups=self.conv_group, kernel_size=3, stride=1, padding=1),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.channel_num),
                                )
        self.fc_restore = nn.Sequential(
                                    nn.Linear(self.data_length, self.data_length),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.data_length),
                                    nn.Linear(self.data_length, self.data_length)
                                )

    def forward(self, de_noise):
        """
        数据解压缩：
        残差（分组卷积） ----> 残差（深度可分离卷积）-----> 残差（深度可分离卷积） -----> 全连接恢复
        :param de_noise: [batch_size, 32*32]
        :return: [batch_size, 32*32]
        """
        # 深度卷积
        out = self.res_unit(self.deep_conv, de_noise)  # [batch_size, 32, 32] -----> [batch_size, 32 ,32]
        # 分组卷积
        out = self.res_unit(self.group_conv1, out)  # [batch_size, 32, 32] -----> [batch_size, 32 ,32]
        # 深度可分卷积
        out = self.res_unit(self.deep_separate, out)  # [batch_size, 32, 32] -----> [batch_size, 32 ,32]
        # 深度可分卷积
        out = self.res_unit(self.deep_separate_2, out)  # [batch_size, 32, 32] ----->[batch_size, 32 ,32]
        # 全连接
        out = out.view(-1, self.data_length)  # [batch_size, 32, 32] -----> [batch_size, 32*32]
        output = self.fc_restore(out)  # [batch_size, 32*32] -----> [batch_size, 32*32]
        return output


class Noise(BaseModel):
    """
    处理噪声：
    1.制造一个高斯白噪声，与encoder_output相加,作为输入Decoder的输入：through_channel
    2.through_channel经过网络，将噪声弱化
    """

    def __init__(self, snr, ratio):
        super(Noise, self).__init__()
        self.snr = snr
        self.ratio = ratio
        self.sub_fc = nn.Sequential(
                                    nn.Linear(int(self.data_length / self.ratio), self.data_length),
                                    nn.ELU(True),
                                    nn.BatchNorm1d(self.data_length),
                                )

    def forward(self, encoder_output):
        """
        加入噪声 -----> 标准化 -----> 全连接层
        :param encoder_output: [batch_size, 32*32/ratio]
        :return: [batch_size, 32, 32]
        """
        # 经过信道，加入噪声
        if self.snr is not None:
            through_channel = self.wgn(encoder_output)
        else:
            through_channel = encoder_output
        # 标准化
        out = self.normalize(through_channel)  # [batch_size, 32*32]
        # 全连接
        out = self.sub_fc(out)  # [batch_size, 32*32]
        out = out.view(-1, self.channel_num, self.channel_num)  # [batch_size, 32, 32]
        return out

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


if __name__ == '__main__':
    model = Seq2Seq(5, 8).to(config.device)
    summary(model, (1024,))
    # for name, param in model.named_parameters():
    #     print(name, param.size())
