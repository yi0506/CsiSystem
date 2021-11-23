"""主函数"""
from libs import config
from libs.utils import v_loop_wrapper, model_snr_loop_wrapper
from libs.RM_net import RMNetConfiguration
from libs.CSI_net import CSINetConfiguration
from libs.cs import CSConfiguration, SAMPCS, OMPCS, SPCS, DCTCS, FFTCS
from libs.visual import CSPlot, HighSpeedNetPlot, TimeForm
from libs.CSI_Frame import RMNet_CSI, CSINet_CSI, CS_CSI


class Xun_Lian_He_Ce_Shi(object):

    def net_csi(self):
        rmnet = RMNet_CSI()
        csinet = CSINet_CSI()

        rmnet.net_joint_train()
        rmnet.net_joint_test()
        csinet.net_joint_train()
        csinet.net_joint_test()

    def cs_csi(self):
        cs = CS_CSI()
        cs.cs_register(OMPCS, CSConfiguration.sparse_dct, CSConfiguration.omp)
        cs.cs_register(SAMPCS, CSConfiguration.sparse_fft, CSConfiguration.samp)
        cs.CS_joint_test()


class Tu_He_Biao(object):

    @v_loop_wrapper
    @model_snr_loop_wrapper
    def all_plot(self, v, criteria, model_snr, img_format="svg", ratio_list=config.ratio_list):
        self.one_plot(v, criteria, model_snr, img_format, ratio_list)

    def one_plot(self, v, criteria, model_snr, img_format="svg", ratio_list=config.ratio_list):
        """绘制图像"""
        networks = [
            RMNetConfiguration.network_name,
            CSINetConfiguration.network_name
        ]
        methods = [
            "{}-{}".format(CSConfiguration.sparse_dct, CSConfiguration.omp),
            "{}-{}".format(CSConfiguration.sparse_dct, CSConfiguration.samp)
        ]
        HighSpeedNetPlot().net_plot(v, networks, criteria, model_snr, img_format, ratio_list)
        CSPlot().cs_plot(v, methods, criteria, img_format, ratio_list)

    @v_loop_wrapper
    def all_form(self, v, ratio_list=config.ratio_list, model_snr=None):
        self.one_form(v, ratio_list, model_snr)

    def one_form(self, v, ratio_list=config.ratio_list, model_snr=None):
        """输出系统耗时json文件，cs、old_csi、net之间的对比"""
        networks = [
            RMNetConfiguration.network_name,
            CSINetConfiguration.network_name
        ]
        methods = [
            "{}-{}".format(CSConfiguration.sparse_dct, CSConfiguration.omp),
            "{}-{}".format(CSConfiguration.sparse_dct, CSConfiguration.samp)
        ]
        tf = TimeForm()
        tf.csi_time_form(v, methods, networks, ratio_list, model_snr)


x1 = Xun_Lian_He_Ce_Shi()
y1 = Tu_He_Biao()
x1.net_csi()
x1.cs_csi()
y1.all_form()
y1.all_plot()
