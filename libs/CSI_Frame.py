"""网络模型与CS模块各项功能的封装"""
import re
import pickle
from libs.visual import main_plot, main_form
import types
import warnings

from libs.train import concurrent_train, train_model_separated, train_model_merged, rec_mkdir
from libs.test import concurrent_test, test_model_separated, test
from libs.cs import CS
from libs import config
from libs.old_csi_net import Tester, Trainer


def v_loop(func):
    """
    在不同速度下进行测试

    v_list为一个列表，则按照列表中的速度进行测试，默认列表中只有config.velocity

    __ratio_loop ----> call_func：     装饰后新的__ratio_loop指向call_func
    func ---->   __ratio_loop：      func指向装饰前原来的__ratio__loop
    call_func(obj, *args, **kwargs) <==> 对象.__ratio_loop(*args, **kwargs)或 self.__ratio_loop(*args, **kwargs)
    call_func(obj, *args, **kwargs) <==> 对象.visual_display(*args, **kwargs)
    obj ---> MyCsi()

    call_func需要接收obj对象的引用是由于：

        (1)在类的外部，通过对象.xxx() 调用了xxx方法（本质是直接调用call_func），此时，会自动传入一个实例对象的引用,使得call_func多接收一个参数, 如:
            csi = MyCsi()
            csi.visual_display()

        (2)在类中，通过self.xxx 调用了xxx方法（本质是直接调用call_func），此时，会自动传入self参数，使得call_func多接收一个参数, 如:
            self.__ratio_loop(concurrent_train, epoch=epoch, model_list=model_list, multi=multi)

    func()调用时需要传入obj对象的引用是由于：
        func指向的是一个方法，xxx方法在定义时为：def xxx(self, )，需要一个参数，接收对象的引用，
        在这里，即装饰器内部，func直接指向该方法，并直接调用了func指向的函数，而不是通过 对象.func()或self.func()的方式调用，
        因此func需要传入一个额外的对象引用obj
    """
    def call_func(obj, *args, **kwargs):
        velocity = kwargs.get('velocity')
        if velocity is not None:
            del kwargs['velocity']
            warnings.warn('velocity不需要传递，请在实例化MyCsi对象时设置v_list，设置不同的velocity', UserWarning)
        for v in obj.v_list:
            func(obj, *args, velocity=v, **kwargs)
    return call_func


class BaseCSIFrame(object):
    """
    定义CSI反馈系统的必要方法：模型训练、测试、传统算法测试、测试结果可视化
    所有代码中（注释、变量名、文件名）提到的old_csi、old均指最早的csi_net网络，csi_net均指model.py中的神经网络模型
    必要参数可参考config.py
    """
    __instance = None  # 是否已创建实例对象
    __initialized = False  # 是否已初始化

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, **kwargs):
        """初始化实例属性"""
        if self.__initialized:
            return
        self.SNRs = kwargs.get("SNRs", config.SNRs)  # 信噪比
        self.ratio_list = kwargs.get("ratio_list", config.ratio_list)  # csi压缩率列表
        self.v_list = kwargs.get("v_list", config.velocity_list)  # 测试速度列表
        assert isinstance(self.ratio_list, list) and isinstance(self.v_list, list) and isinstance(self.SNRs, list), "请输入一个列表"
        self.__cs_data_length = config.cs_data_length  # data长度
        self.__test_list = list()  # CS方法
        for method in config.method_list:
            ret = re.match(r"^([a-zA-Z]+):([a-zA-Z]+_?[a-zA-Z]+)$", method)
            if ret:
                self.__test_list.append(ret)
        self.__initialized = True  # 已执行初始化

    @v_loop
    def ratio_loop(self, func, *args, **kwargs):
        """对func传入所需要的压缩率ratio"""
        for ratio in self.ratio_list:
            if isinstance(func, types.FunctionType) or isinstance(func, types.MethodType):
                func(*args, ratio=ratio, **kwargs)
            elif isinstance(func, type):
                func(*args, ratio=ratio, **kwargs)()
            else:
                print('变量类型有问题')
                return

    @v_loop
    def visual_display(self,
                       velocity,
                       train_models=config.train_model_SNRs,
                       cs_multi_ratio_model=None,
                       cs_multi_ratio_list=None,
                       form_model=None,
                       ):
        """
        测试结果进行可视化展示，

        Parameter:

            train_models -> 选择进行绘图的模型
            velocity -> 选择的速度，velocity不需要传递，由MyCsi实例对象的v_list属性决定
            cs_multi_ratio_model -> 传统CS算法绘图，多压缩率合并，选择何种的网络模型进行比较，默认选择NonedB模型
            cs_multi_ratio_list -> 传统CS算法绘图，多压缩率合并，选择何种的网络模型进行比较，默认选择NonedB模型

        """
        if cs_multi_ratio_list is None:
            cs_multi_ratio_list = [2, 8]
        main_plot(train_models=train_models, velocity=velocity, cs_multi_ratio_list=cs_multi_ratio_list,
                  ratio_list=self.ratio_list, cs_multi_ratio_model=cs_multi_ratio_model)
        main_form(velocity=velocity, ratio_list=self.ratio_list, model=form_model)

    def model_joint_train(self, epoch=300, model_list=config.train_model_SNRs, multi=True) -> None:
        """在不同压缩率、使用多进程，训练不同信噪比的模型"，seq2seq与noise合并在一起，模型会保存成一个整体"""
        pass

    def model_joint_test(self, model_list=config.train_model_SNRs, multi=True) -> None:
        """在不同压缩率、不同信噪比下，使用多进程，测试不同信噪比模型的效果，测试的是合并在一起的整体模型"""
        pass

    def model_train(self, epoch=300, snr=None) -> None:
        """snr: 训练某种信噪比下的模型， 在不同压缩率下，进行训练，训练整个神经网络，模型会保存成一个整体"""
        pass

    def model_test(self, snr=None) -> None:
        """snr：选择某种信噪比模型，在不同压缩率下，在不同信噪比下，测试整体模型的效果，即在哪种信噪比下训练的，就测试哪种信噪比"""
        pass

    def model_train_separated(self, snr: int or None = None, epoch=200) -> None:
        """snr：指定某种信噪比，在不同压缩率下，训练指定的模型，seq2seq与noise合并在一起训练，
        但encoder ,noise, decoder三个模块分开保存成三个模型，snr为指定的信噪比"""
        pass

    def model_test_separated(self, snr: int or None = None) -> None:
        """在不同压缩率下，测试分开的三个模型（encoder + decoder + noise）在不同信噪比下的效果，
        snr表示选择哪一个信噪比下的模型，且要求相应的信噪比模型必须存在，否则找不到该模型"""
        pass

    def cs_test(self) -> None:
        """在不同信噪比下对所有CS算法的评估，并将结果保存到文件中"""
        pass

    def old_csi_train(self, epoch: int = 1000) -> None:
        """old_csi_net在不同压缩率下模型的训练"""
        pass

    def old_csi_test(self, add_noise: bool = True) -> None:
        """
        old_csi_net模型在不同压缩率、不同信噪比下测试，并将测试结果保存
        :param add_noise: 是否加入噪声
        """
        pass


class MyCsi(BaseCSIFrame):
    """
    完成CSI反馈系统的必要方法：模型训练、测试、传统算法测试、测试结果可视化
    """

    def model_joint_train(self, epoch=300, model_list=config.train_model_SNRs, multi=True):
        self.ratio_loop(concurrent_train, epoch=epoch, model_list=model_list, multi=multi)

    def model_joint_test(self, model_list=config.train_model_SNRs, multi=True):
        self.ratio_loop(concurrent_test, multi=multi, model_list=model_list)

    def model_train(self, epoch=300, snr=None) -> None:
        self.ratio_loop(train_model_merged, epoch=epoch, snr=snr)

    def model_test(self, snr=None) -> None:
        self.ratio_loop(test, snr_model=snr)

    def model_train_separated(self, snr=None, epoch=200):
        self.ratio_loop(train_model_separated, epoch=epoch, snr=snr)

    def model_test_separated(self, snr=None):
        self.ratio_loop(test_model_separated, snr=snr)

    def cs_test(self) -> None:
        self.ratio_loop(self.cs_test_done)

    def old_csi_train(self, epoch=1000):
        self.ratio_loop(Trainer, epoch=epoch)

    def old_csi_test(self, add_noise=True):
        self.ratio_loop(self.old_csi_test_done, add_noise=add_noise)

    def old_csi_test_done(self, add_noise, velocity, ratio):
        """odl_csi执行测试"""
        data_dict = dict()
        if add_noise is True:
            for snr in self.SNRs:
                model = Tester(add_noise=add_noise, snr=snr, ratio=ratio, velocity=velocity)
                result = model.run()
                data_dict["{}dB".format(snr)] = result
                file_path = "./test_result/{}km/old_csi/old_csi_{}.pkl".format(velocity, ratio)
                rec_mkdir(file_path)
                pickle.dump(data_dict, open(file_path, "wb"))
                print(data_dict)
        else:
            model = Tester()
            result = model.run()
            data_dict["no_noise"] = result
            file_path = "./test_result/{}km/old_csi/old_csi_no_noise_{}.pkl".format(velocity, ratio)
            rec_mkdir(file_path)
            pickle.dump(data_dict, open(file_path, "wb"))
            print(data_dict)

    def cs_test_done(self, velocity, ratio):
        """cs执行测试"""
        Fi_m = int(self.__cs_data_length / ratio)
        cs_dict = {
            "k": config.k,
            "t": config.t,
            "Fi_m": Fi_m,
            "Fi_ratio": ratio,
            "velocity": velocity
        }
        # 对每种方法在不同信噪比下进行测试
        for temp in self.__test_list:
            result_dict = dict()
            sparse = temp.group(1).lower()
            restore = temp.group(2).lower()
            for snr in self.SNRs:
                if restore == "idct" or restore == "ifft":
                    result = CS(snr=snr, sparse_method=sparse, restore_method=restore, full_sampling=True, **cs_dict)()
                else:
                    result = CS(snr=snr, sparse_method=sparse, restore_method=restore, full_sampling=False, **cs_dict)()
                result_dict["{}dB".format(snr)] = result
            # 保存测试结果到文件中
            file_name = """{}-{}.pkl""".format(restore, ratio)
            file_path = "./test_result/{}km/cs/ratio_{}/result/".format(velocity, ratio) + file_name
            rec_mkdir(file_path)
            with open(file_path, "wb") as f:
                pickle.dump(result_dict, f)


if __name__ == '__main__':
    pass
