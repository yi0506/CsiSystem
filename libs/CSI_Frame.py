"""网络模型与CS模块各项功能的封装"""
import pickle
import torch
from libs.RM_net import RMNetConfiguration
from libs.CSI_net import CSINetConfiguration

from utils import ratio_loop_wrapper, model_snr_loop_wrapper, v_loop_wrapper
import config
from RM_net import RMNet
from CSI_net import CsiNet
from utils import rec_mkdir, SingletonType, train, test
from csi_dataset import CSDataset, RMNetDataset, CSINetDataset


class High_Speed_Net_CSI(metaclass=SingletonType):
    """通用神经网络 CSI执行"""
    CSI_MODEL = None  # 执行CSI的模型
    CSI_DATASET = None  # 执行CSI的模型的数据集
    NETWORK_NAME = None  # 网络模型名称

    def net_train(self, ratio, v, model_snr, epoch=config.epoch) -> None:
        """在不同压缩率下，进行训练某个信噪比的模型"""
        model = self.CSI_MODEL(ratio)
        save_path = "./model/{}km/{}/ratio_{}/{}_{}dB.ckpt".format(v, self.NETWORK_NAME, ratio, self.NETWORK_NAME, model_snr)
        info = "v:{}\tratio:{}".format(v, ratio)
        dataloader = self.CSI_DATASET(True, v).get_data_loader()
        train(model, epoch, save_path, dataloader, model_snr, info)

    @model_snr_loop_wrapper
    @v_loop_wrapper
    @ratio_loop_wrapper
    def net_joint_train(self, ratio, v, model_snr, epoch=config.epoch) -> None:
        """在不同压缩率、不同速度的信道模型下，训练不同信噪比的模型"""
        self.net_train(ratio, v, model_snr, epoch)

    def net_test(self, ratio, v, model_snr, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在某个压缩率，某个速度下，测试不同信噪比下模型的效果"""
        model = self.CSI_MODEL(ratio)
        model_path = "./model/{}km/{}/ratio_{}/{}_{}dB.ckpt".format(v, self.NETWORK_NAME, ratio, self.NETWORK_NAME, model_snr)
        model.load_state_dict(torch.load(model_path), strict=False)
        info = "v:{}\tratio:{}".format(v, ratio)
        data_loader = self.CSI_DATASET(False, v).get_data_loader()
        result_dict = dict()
        for snr in SNRs:
            result_dict["{}dB".format(snr)] = test(model, data_loader, snr, info)
        del model
        if save_ret:
            save_path = "./test_result/{}km/{}/ratio_{}/{}_{}dB.pkl".format(v, self.NETWORK_NAME, ratio, self.NETWORK_NAME, model_snr) if not save_path else save_path
            rec_mkdir(save_path)
            pickle.dump(result_dict, open(save_path, "wb"))

    @model_snr_loop_wrapper
    @v_loop_wrapper
    @ratio_loop_wrapper
    def net_joint_test(self, ratio, v, model_snr, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在不同压缩率、不同速度的信道模型下，测试不同信噪比模型的效果"""
        self.net_test(ratio, v, model_snr, SNRs, save_ret, save_path)


class RMNet_CSI(High_Speed_Net_CSI):
    """RM_net CSI执行"""
    CSI_MODEL = RMNet
    CSI_DATASET = RMNetDataset
    NETWORK_NAME = RMNetConfiguration.network_name


class CSINet_CSI(High_Speed_Net_CSI):
    """csi net 执行CSI"""
    CSI_MODEL = CsiNet
    CSI_DATASET = CSINetDataset
    NETWORK_NAME = CSINetConfiguration.network_name


class CS_CSI(metaclass=SingletonType):
    """CS 执行CSI"""
    RESTORE_DIC = dict()
    CS_TP = list()
    DATASET = CSDataset
    
    def cs_register(self, cls, sparse, restore):
        """
        注册CS的方法

        cls: 
        sparse: 采用的稀疏基，"dct" 或者 "fft"
        restore: 采用的恢复方法 

        """
        self.CS_TP.append((sparse, restore))
        self.RESTORE_DIC[restore] = cls

    def CS_test(self, v, ratio, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在不同信噪比下对所有CS算法的评估，并将结果保存到文件中"""
        data_loader = CSDataset(v).get_data_loader()
        for sparse, restore in self.CS_TP:
            result_dict = dict()
            full_sampling = False
            if restore == "idct" or restore == "ifft":
                full_sampling = True
            # 在不同信噪比下测试CS方法的效果
            for snr in SNRs:
                ret = self.RESTORE_DIC[restore](snr=snr, ratio=ratio, v=v, sparse=sparse, restore=restore, full_sampling=full_sampling, data_loader=data_loader)()
                result_dict["{}dB".format(snr)] = ret
                
            if save_ret:
                # 保存测试结果到文件中
                save_path = "./test_result/{}km/cs/ratio_{}/result/{}-{}.pkl".format(v, ratio, sparse, restore) if not save_path else save_path
                rec_mkdir(save_path)
                with open(save_path, "wb") as f:
                    pickle.dump(result_dict, f)

    @v_loop_wrapper
    @ratio_loop_wrapper
    def CS_joint_test(self, v, ratio, SNRs=config.SNRs, save_ret: bool = True, save_path: str = ""):
        self.CS_test(v, ratio, SNRs, save_ret, save_path)


if __name__ == '__main__':
    pass
