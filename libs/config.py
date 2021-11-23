"""CSI系统各项参数配置文件"""

import torch
import os


####################### 系统 #############################
Nt = 16  # 发射天线数目
velocity_list = [50, 100, 150, 200, 300]  # 速度集合，单位km/h [10km/h 50km/h 100km/h 150km/h 200km/h 300km/h]
ratio_list = [2, 4, 8, 16, 32]  # 压缩率列表
y_ticks_similarity = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 画图时，相似度的y轴
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
SNRs = [-10, -8, -6, -4, -2, 0, 5, 10, 15, 20]  # 测试时的信噪比，画图的横坐标不能有None
model_SNRs = [None]  # 不同信噪比的模型，None表示无噪声



################ network #################
train_batch_size = 250
test_batch_size = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clip = 0.01  # 梯度裁剪阈值
epoch = 20


if __name__ == '__main__':
    print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
