from sklearn.decomposition import PCA
import numpy as np
import torch
from torch.nn.functional import mse_loss


def pca_process(ratio, data):
    """
    PCA压缩与恢复
    :param old_data: 二维矩阵
    :return:
    """
    loss_list = list()
    similarity_list = list()

    # 实例化pca
    pca = PCA(n_components=ratio)

    # PCA压缩
    compressed_data = pca.fit_transform(data)
    print(compressed_data.shape)

    # pca重构
    restore_data = np.dot(compressed_data, pca.components_) + pca.mean_

    # 计算准确率与损失
    data = torch.tensor(data)
    restore_data = torch.tensor(restore_data)
    ret_similarity = torch.cosine_similarity(data, restore_data, dim=-1).mean()
    ret_loss = mse_loss(data, restore_data)
    similarity_list.append(ret_similarity)
    loss_list.append(ret_loss)
    print({"similarity": np.mean(similarity_list), "loss": np.mean(loss_list)})


def dataset(num=40):
    for i in range(num):
        yield np.random.randn(200, 1024)


if __name__ == "__main__":
    data_loader = dataset()
    pca_process(0.9, next(data_loader))
