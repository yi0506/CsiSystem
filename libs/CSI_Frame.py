"""网络模型与CS模块各项功能的封装"""
import pickle
import torch

from utils import ratio_loop_wrapper, v_loop_wrapper
import config
from RM_net import RMNet, RMNetConfiguration
from CSI_net import CsiNet, CSINetConfiguration
from CSP_net import CSPNet, CSPNetConfiguration
from CSI_net_stu import CSINetStuConfiguration, CSINetStu
from ISTA_Net_Plus import ISTANetplus, ISTANetplusConfiguration
from ISTA_Net import ISTANet, ISTANetConfiguration
from FISTA_Net import FISTANet, FISTANetConfiguration
from TD_FISTA_Net import TDFISTANet, TDFISTANetConfiguration
from utils import rec_mkdir, SingletonType
from libs.test import csp_test, test, comm_csi_test, ista_test, td_fista_test, fista_test
from libs.train import fista_train, td_fista_train, train, train_stu, csp_train, ista_train
from csi_dataset import HS_RMNetDataset, HS_CSINetDataset, HS_RMStuNetDataset, CSPNetDataset, HS_CSINetStuDataset, HS_CSDataset
from csi_dataset import COMM_CSDataset,  COMM_CSINetDataset, COMM_CSINetStuDataset, COMM_RMNetDataset, COMM_RMNetStuDataset
from csi_dataset import COMM_ISTANet_Dataset, COMM_ISTANet_Plus_Dataset, COMM_FISTANet_Dataset, COMM_TD_FISTANet_Dataset
from RMNet_stu import RMNetStu, RMNetStuConfiguration


class COMM_Net_CSI(metaclass=SingletonType):
    """一般网络CSI 执行"""
    CSI_MODEL = None  # 执行CSI的模型
    CSI_DATASET = None  # 执行CSI的模型的数据集
    NETWORK_NAME = ""  # 网络模型名称
    TRAIN_FUNC = train  # 网络训练函数指针
    TEST_FUNC = test  # 网络测试函数指针

    def net_train(self, ratio, epoch=config.epoch, save_path: str = "") -> None:
        """在不同压缩率下，进行训练某个信噪比的模型"""
        model = self.CSI_MODEL(ratio)
        save_path = "./model/{}/common/ratio_{}/{}.ckpt".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME) if not save_path else save_path
        info = "{}:\tratio:{}".format(self.NETWORK_NAME, ratio)
        dataloader = self.CSI_DATASET(True).get_data_loader()
        self.TRAIN_FUNC(model, epoch, save_path, dataloader, info)

    @ratio_loop_wrapper
    def net_joint_train(self, ratio, epoch=config.epoch) -> None:
        """在不同压缩率、不同速度的信道模型下，训练不同信噪比的模型"""
        self.net_train(ratio, epoch)

    def net_test(self, ratio, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在某个压缩率，测试不同信噪比下模型的效果"""
        model = self.CSI_MODEL(ratio)
        model_path = "./model/{}/common/ratio_{}/{}.ckpt".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME)
        model.load_state_dict(torch.load(model_path), strict=False)
        info = "{}:\tratio:{}".format(self.NETWORK_NAME, ratio)
        data_loader = self.CSI_DATASET(False).get_data_loader()
        result_dict = dict()
        for snr in SNRs:
            result_dict["{}dB".format(snr)] = self.TEST_FUNC(model, data_loader, snr, info)
        del model
        if save_ret:
            save_path = "./test_result/{}/common/ratio_{}/{}.pkl".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME) if not save_path else save_path
            rec_mkdir(save_path)
            pickle.dump(result_dict, open(save_path, "wb"))

    @ratio_loop_wrapper
    def net_joint_test(self, ratio, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在不同压缩率,测试不同信噪比模型的效果"""
        self.net_test(ratio, SNRs, save_ret, save_path)


class ISTANet_CSI(COMM_Net_CSI):
    """ISTANet CSI执行"""
    CSI_MODEL = ISTANet  # 执行CSI的模型
    CSI_DATASET = COMM_ISTANet_Dataset  # 执行CSI的模型的数据集
    NETWORK_NAME = ISTANetConfiguration.network_name  # 网络模型名称
    TRAIN_FUNC = ista_train
    TEST_FUNC = ista_test

    def net_train(self, ratio, layer_num=ISTANetConfiguration.layer_num, epoch=config.epoch, save_path: str = "") -> None:
        """在不同压缩率下，进行训练某个信噪比的模型"""
        model = self.CSI_MODEL(layer_num)
        save_path = "./model/{}/common/ratio_{}/{}_{}.ckpt".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME, layer_num) if not save_path else save_path
        info = "{}:\tratio:{}".format(self.NETWORK_NAME, ratio)
        dataset = self.CSI_DATASET(True, ratio)
        dataloader = dataset.get_data_loader()
        self.TRAIN_FUNC(model, epoch, dataset.Qinit, dataset.Phi, layer_num, save_path, dataloader, info)

    def net_test(self, ratio, layer_num=ISTANetConfiguration.layer_num, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在某个压缩率，测试不同信噪比下模型的效果"""
        model = self.CSI_MODEL(layer_num)
        model_path = "./model/{}/common/ratio_{}/{}_{}.ckpt".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME, layer_num)
        model.load_state_dict(torch.load(model_path))
        info = "{}:\tratio:{}".format(self.NETWORK_NAME, ratio)
        dataset = self.CSI_DATASET(False, ratio)
        dataloader = dataset.get_data_loader()
        result_dict = dict()
        for snr in SNRs:
            result_dict["{}dB".format(snr)] = self.TEST_FUNC(model, dataset.Phi, dataset.Qinit, dataloader, snr, info)
        del model
        if save_ret:
            save_path = "./test_result/{}/common/ratio_{}/{}.pkl".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME) if not save_path else save_path
            rec_mkdir(save_path)
            pickle.dump(result_dict, open(save_path, "wb"))


class ISTANetPlus_CSI(ISTANet_CSI):
    """ISTANetPlus CSI执行"""
    CSI_MODEL = ISTANetplus  # 执行CSI的模型
    CSI_DATASET = COMM_ISTANet_Plus_Dataset  # 执行CSI的模型的数据集
    NETWORK_NAME = ISTANetplusConfiguration.network_name  # 网络模型名称
    TRAIN_FUNC = ista_train
    TEST_FUNC = ista_test


class FISTANet_CSI(ISTANet_CSI):
    """ISTANetPlus CSI执行"""
    CSI_MODEL = FISTANet  # 执行CSI的模型
    CSI_DATASET = COMM_FISTANet_Dataset  # 执行CSI的模型的数据集
    NETWORK_NAME = FISTANetConfiguration.network_name  # 网络模型名称
    TRAIN_FUNC = fista_train
    TEST_FUNC = fista_test


class TD_FISTANet_CSI(COMM_Net_CSI):
    """ISTANetPlus CSI执行"""
    CSI_MODEL = TDFISTANet  # 执行CSI的模型
    CSI_DATASET = COMM_TD_FISTANet_Dataset  # 执行CSI的模型的数据集
    NETWORK_NAME = TDFISTANetConfiguration.network_name  # 网络模型名称
    TRAIN_FUNC = td_fista_train
    TEST_FUNC = td_fista_test

    def net_train(self, ratio, layer_num=TDFISTANetConfiguration.layer_num, epoch=config.epoch, save_path: str = "") -> None:
        """在不同压缩率下，进行训练某个信噪比的模型"""
        model = self.CSI_MODEL(layer_num, ratio)
        save_path = "./model/{}/common/ratio_{}/{}_{}.ckpt".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME, layer_num) if not save_path else save_path
        info = "{}:\tratio:{}".format(self.NETWORK_NAME, ratio)
        dataset = self.CSI_DATASET(True, ratio)
        dataloader = dataset.get_data_loader()
        self.TRAIN_FUNC(model, epoch, dataset.Qinit, layer_num, save_path, dataloader, info)

    def net_test(self, ratio, layer_num=TDFISTANetConfiguration.layer_num, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在某个压缩率，测试不同信噪比下模型的效果"""
        model = self.CSI_MODEL(layer_num, ratio)
        model_path = "./model/{}/common/ratio_{}/{}_{}.ckpt".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME, layer_num)
        model.load_state_dict(torch.load(model_path))
        info = "{}:\tratio:{}".format(self.NETWORK_NAME, ratio)
        dataset = self.CSI_DATASET(False, ratio)
        dataloader = dataset.get_data_loader()
        result_dict = dict()
        for snr in SNRs:
            result_dict["{}dB".format(snr)] = self.TEST_FUNC(model, dataset.Qinit, dataloader, snr, info)
        del model
        if save_ret:
            save_path = "./test_result/{}/common/ratio_{}/{}.pkl".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME) if not save_path else save_path
            rec_mkdir(save_path)
            pickle.dump(result_dict, open(save_path, "wb"))


class CSPNet_CSI(COMM_Net_CSI):
    """CSPNet CSI执行"""
    CSI_MODEL = CSPNet  # 执行CSI的模型
    CSI_DATASET = CSPNetDataset  # 执行CSI的模型的数据集
    NETWORK_NAME = CSPNetConfiguration.network_name  # 网络模型名称
    TRAIN_FUNC = csp_train
    TEST_FUNC = csp_test

    def net_train(self, ratio, epoch=config.epoch, save_path: str = "") -> None:
        """在不同压缩率下，进行训练某个信噪比的模型"""
        model = self.CSI_MODEL(ratio)
        save_path = "./model/{}/common/ratio_{}/{}.ckpt".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME) if not save_path else save_path
        info = "{}:\tratio:{}".format(self.NETWORK_NAME, ratio)
        dataset = self.CSI_DATASET(True, ratio)
        dataloader = dataset.get_data_loader()
        self.TRAIN_FUNC(model, epoch, dataset.Phi, save_path, dataloader, info)

    def net_test(self, ratio, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在某个压缩率，测试不同信噪比下模型的效果"""
        model = self.CSI_MODEL(ratio)
        model_path = "./model/{}/common/ratio_{}/{}.ckpt".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME)
        model.load_state_dict(torch.load(model_path), strict=False)
        info = "{}:\tratio:{}".format(self.NETWORK_NAME, ratio)
        dataset = self.CSI_DATASET(False, ratio)
        data_loader = dataset.get_data_loader()
        result_dict = dict()
        for snr in SNRs:
            result_dict["{}dB".format(snr)] = self.TEST_FUNC(model, dataset.Phi, data_loader, snr, info)
        del model
        if save_ret:
            save_path = "./test_result/{}/common/ratio_{}/{}.pkl".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME) if not save_path else save_path
            rec_mkdir(save_path)
            pickle.dump(result_dict, open(save_path, "wb"))


class COMM_CSINet_CSI(COMM_Net_CSI):
    """CSINet CSI执行"""
    CSI_MODEL = CsiNet  # 执行CSI的模型
    CSI_DATASET = COMM_CSINetDataset  # 执行CSI的模型的数据集
    NETWORK_NAME = CSINetConfiguration.network_name  # 网络模型名称
    TEST_FUNC = comm_csi_test  # 网络测试函数指针


class COMM_RMNet_CSI(COMM_Net_CSI):
    """CSINet CSI执行"""
    CSI_MODEL = RMNet  # 执行CSI的模型
    CSI_DATASET = COMM_RMNetDataset  # 执行CSI的模型的数据集
    NETWORK_NAME = RMNetConfiguration.network_name  # 网络模型名称
    TEST_FUNC = comm_csi_test  # 网络测试函数指针


class COMM_NetStu_CSI(COMM_Net_CSI):
    """学生模型 CSI执行"""
    TEACHER = None
    TRAIN_FUNC = train_stu

    def net_train(self, ratio, epoch=config.epoch, save_path: str = "") -> None:
        """在不同压缩率下，进行训练某个信噪比的模型"""
        stu = self.CSI_MODEL(ratio)
        teacher = self.TEACHER.CSI_MODEL(ratio)
        teacher_path = "./model/{}/common/ratio_{}/{}.ckpt".format(self.TEACHER.NETWORK_NAME, ratio, self.TEACHER.NETWORK_NAME)
        teacher.load_state_dict(torch.load(teacher_path))
        save_path = "./model/{}/common/ratio_{}/{}.ckpt".format(self.NETWORK_NAME, ratio, self.NETWORK_NAME) if not save_path else save_path
        info = "{}\tratio:{}".format(self.NETWORK_NAME, ratio)
        dataloader = self.CSI_DATASET(True).get_data_loader()
        self.TRAIN_FUNC(teacher, stu, epoch, save_path, dataloader, info)


class COMM_CSINetStu_CSI(COMM_NetStu_CSI):
    """CSINetStu CSI执行"""
    CSI_MODEL = CSINetStu
    CSI_DATASET = COMM_CSINetStuDataset  # RMStuNet为学生模型，与教师模型使用同样数据集
    NETWORK_NAME = CSINetStuConfiguration.network_name
    TEACHER = COMM_CSINet_CSI


class COMM_RMNetStu_CSI(COMM_NetStu_CSI):
    """CSINetStu CSI执行"""
    CSI_MODEL = RMNetStu
    CSI_DATASET = COMM_RMNetStuDataset  # RMStuNet为学生模型，与教师模型使用同样数据集
    NETWORK_NAME = RMNetStuConfiguration.network_name
    TEACHER = COMM_RMNet_CSI


class COMM_CS_CSI(metaclass=SingletonType):
    """CS 执行CSI"""
    RESTORE_DIC = dict()
    CS_TP = list()
    DATASET = COMM_CSDataset

    def cs_register(self, cls, sparse, restore):
        """
        注册CS的方法

        cls: CS类
        sparse: 采用的稀疏基，"dct" 或者 "fft"
        restore: 采用的恢复方法

        """
        self.CS_TP.append((sparse, restore))
        self.RESTORE_DIC[restore] = cls

    def cs_test(self, ratio, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在不同信噪比下对所有CS算法的评估，并将结果保存到文件中"""
        data_loader = COMM_CSDataset().get_data_loader()
        for sparse, restore in self.CS_TP:
            result_dict = dict()
            full_sampling = False
            if restore == "idct" or restore == "ifft":
                full_sampling = True
            # 在不同信噪比下测试CS方法的效果
            for snr in SNRs:
                ret = self.RESTORE_DIC[restore](snr=snr, ratio=ratio, sparse=sparse, restore=restore, full_sampling=full_sampling, data_loader=data_loader)()
                result_dict["{}dB".format(snr)] = ret
            if save_ret:
                # 保存测试结果到文件中
                save_path = "./test_result/cs/common/ratio_{}/{}-{}.pkl".format(ratio, sparse, restore) if not save_path else save_path
                rec_mkdir(save_path)
                with open(save_path, "wb") as f:
                    pickle.dump(result_dict, f)

    @ratio_loop_wrapper
    def CS_joint_test(self, ratio, SNRs=config.SNRs, save_ret: bool = True, save_path: str = ""):
        self.cs_test(ratio, SNRs, save_ret, save_path)


class High_Speed_Net_CSI(metaclass=SingletonType):
    """高速环境下神经网络 CSI执行"""
    CSI_MODEL = None  # 执行CSI的模型
    CSI_DATASET = None  # 执行CSI的模型的数据集
    NETWORK_NAME = None  # 网络模型名称
    TRAIN_FUNC = train  # 网络训练函数指针
    TEST_FUNC = test  # 网络测试函数指针

    def net_train(self, ratio, v, epoch=config.epoch, save_path: str = "") -> None:
        """在不同压缩率下，进行训练某个信噪比的模型"""
        model = self.CSI_MODEL(ratio)
        save_path = "./model/{}/HS/{}km/ratio_{}/{}.ckpt".format(self.NETWORK_NAME, v, ratio, self.NETWORK_NAME) if not save_path else save_path
        info = "{}\t:v:{}\tratio:{}".format(self.NETWORK_NAME, v, ratio)
        dataloader = self.CSI_DATASET(True, v).get_data_loader()
        self.TRAIN_FUNC(model, epoch, save_path, dataloader, info)

    @v_loop_wrapper
    @ratio_loop_wrapper
    def net_joint_train(self, ratio, v, epoch=config.epoch) -> None:
        """在不同压缩率、不同速度的信道模型下，训练不同信噪比的模型"""
        self.net_train(ratio, v, epoch)

    def net_test(self, ratio, v, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在某个压缩率，某个速度下，测试不同信噪比下模型的效果"""
        model = self.CSI_MODEL(ratio)
        model_path = "./model/{}/HS/{}km/ratio_{}/{}.ckpt".format(self.NETWORK_NAME, v, ratio, self.NETWORK_NAME)
        model.load_state_dict(torch.load(model_path), strict=False)
        info = "{}\tv:{}\tratio:{}".format(self.NETWORK_NAME, v, ratio)
        data_loader = self.CSI_DATASET(False, v).get_data_loader()
        result_dict = dict()
        for snr in SNRs:
            result_dict["{}dB".format(snr)] = self.TEST_FUNC(model, data_loader, snr, info)
        del model
        if save_ret:
            save_path = "./test_result/{}/HS/{}km/ratio_{}/{}.pkl".format(self.NETWORK_NAME, v, ratio, self.NETWORK_NAME) if not save_path else save_path
            rec_mkdir(save_path)
            pickle.dump(result_dict, open(save_path, "wb"))

    @v_loop_wrapper
    @ratio_loop_wrapper
    def net_joint_test(self, ratio, v, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在不同压缩率、不同速度的信道模型下，测试不同信噪比模型的效果"""
        self.net_test(ratio, v, SNRs, save_ret, save_path)


class HS_CS_CSI(COMM_CS_CSI):
    """高速移动环境下CS CSI执行"""

    def cs_test(self, v, ratio, SNRs=config.SNRs, save_ret: bool = True, save_path: str = "") -> None:
        """在不同信噪比下对所有CS算法的评估，并将结果保存到文件中"""
        data_loader = HS_CSDataset(v).get_data_loader()
        for sparse, restore in self.CS_TP:
            result_dict = dict()
            full_sampling = False
            if restore == "idct" or restore == "ifft":
                full_sampling = True
            # 在不同信噪比下测试CS方法的效果
            for snr in SNRs:
                ret = self.RESTORE_DIC[restore](snr=snr, ratio=ratio, sparse=sparse, restore=restore, full_sampling=full_sampling, data_loader=data_loader)()
                result_dict["{}dB".format(snr)] = ret
            if save_ret:
                # 保存测试结果到文件中
                save_path = "./test_result/cs/HS/{}km/ratio_{}/{}-{}.pkl".format(v, ratio, sparse, restore) if not save_path else save_path
                rec_mkdir(save_path)
                with open(save_path, "wb") as f:
                    pickle.dump(result_dict, f)

    @v_loop_wrapper
    @ratio_loop_wrapper
    def CS_joint_test(self, v, ratio, SNRs=config.SNRs, save_ret: bool = True, save_path: str = ""):
        self.cs_test(v, ratio, SNRs, save_ret, save_path)


class HS_RMNet_CSI(High_Speed_Net_CSI):
    """RM_net CSI执行"""
    CSI_MODEL = RMNet
    CSI_DATASET = HS_RMNetDataset
    NETWORK_NAME = RMNetConfiguration.network_name


class HS_CSINet_CSI(High_Speed_Net_CSI):
    """csi net 执行CSI"""
    CSI_MODEL = CsiNet
    CSI_DATASET = HS_CSINetDataset
    NETWORK_NAME = CSINetConfiguration.network_name


class HS_NetStu_CSI(High_Speed_Net_CSI):
    """学生模型 CSI执行"""
    TEACHER = None
    TRAIN_FUNC = train_stu

    def net_train(self, ratio, v, epoch=config.epoch, save_path: str = "") -> None:
        """在不同压缩率下，进行训练某个信噪比的模型"""
        stu = self.CSI_MODEL(ratio)
        teacher = self.TEACHER.CSI_MODEL(ratio)
        teacher_path = "./model/{}/HS/{}km/ratio_{}/{}.ckpt".format(self.TEACHER.NETWORK_NAME, v, ratio, self.TEACHER.NETWORK_NAME)
        teacher.load_state_dict(torch.load(teacher_path))
        save_path = "./model/{}/HS/{}km/ratio_{}/{}.ckpt".format(self.NETWORK_NAME, v, ratio, self.NETWORK_NAME) if not save_path else save_path
        info = "{}\tv:{}\tratio:{}".format(self.NETWORK_NAME, v, ratio)
        dataloader = self.CSI_DATASET(True, v).get_data_loader()
        self.TRAIN_FUNC(teacher, stu, epoch, save_path, dataloader, info)


class HS_RMNetStu_CSI(HS_NetStu_CSI):
    """RM_stu_net CSI执行"""
    CSI_MODEL = RMNetStu
    CSI_DATASET = HS_RMStuNetDataset  # RMStuNet为学生模型，与教师模型使用同样数据集
    NETWORK_NAME = RMNetStuConfiguration.network_name
    TEACHER = HS_RMNet_CSI


class HS_CSINetStu_CSI(HS_NetStu_CSI):
    """RM_stu_net CSI执行"""
    CSI_MODEL = CSINetStu
    CSI_DATASET = HS_CSINetStuDataset
    NETWORK_NAME = CSINetStuConfiguration.network_name
    TEACHER = HS_CSINet_CSI


if __name__ == '__main__':
    pass
