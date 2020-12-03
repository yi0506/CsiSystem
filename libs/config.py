"""CSI系统各项参数配置文件"""

import torch


#################### 训练/测试 ################

is_train = True  # 是否使用训练集，data: [batch_size, 32*32]
velocity = 50  # 不同速度下的数据集，单位km/h，[10km/h 50km/h 100km/h 150km/h 200km/h 300km/h]
velocity_list = [50, 100, 150, 200, 300]  # 速度集合
ratio_list = [2, 4, 8, 16, 32]  # 压缩率列表


################ model\dataset #################

shuffle = True  # 是否打乱数据集
net_compress_ratio = 16  # 神经网络，最后全连接层的压缩比率
channel_num = 32  # 信号矩阵通道数
data_length = 32 * 32  # 信号矩阵变成一维向量后的长度
train_batch_size = 250
test_batch_size = 100
SNRs = [-10, -8, -6, -4, -2, 0, 5, 10, 15, 20]  # 测试时的信噪比
train_model_SNRs = [-10, -8, -6, -4, -2, 0, 5, 10, 15, 20, None]  # 不同信噪比的模型
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
clip = 0.01  # 梯度裁剪阈值
conv_group = 16  # 分组卷积的分组数
channel_multiple = 4  # 卷积层通道倍数


###################  CS  ##################

cs_data_length = 32 * 32
k = 32  # k 稀疏度，由于存在sp算法，k不能大于32，否则k会大于Beta矩阵的行数
full_sampling = False  # 是否全采样，即使用dct，fft压缩
t = 1  # samp算法步长
# method_list = ["dct:dct_omp", "dct:dct_samp", "fft:fft_omp", "fft:fft_samp", "dct:idct", "fft:ifft", "dct:dct_sp", "fft:fft_sp"]  # cs方法列表
method_list = ["dct:dct_omp", "fft:fft_omp", "dct:idct", "fft:ifft", "dct:dct_sp", "fft:fft_sp"]  # cs方法列表
# method_list = ["dct:dct_sp"]


################### old_csi ##################
old_csi_net_compress_ratio = 32  # 压缩率
old_csi_data_length = 32 * 32  # 信道矩阵元素个数
old_csi_slope = 0.3  # leaky—relu的负斜率
old_csi_channel_num = 32  # 信道矩阵通道数


if __name__ == '__main__':
    print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
