"""主函数"""
from thop import profile, clever_format
import torch

from libs import config
from libs.utils import criteria_loop_wrapper
from libs.RM_net import RMNetConfiguration, RMNet
from libs.CSI_net import CSINetConfiguration, CsiNet
from libs.RMNet_stu import RMNetStuConfiguration, RMNetStu
from libs.TD_FISTA_Net import TDFISTANetConfiguration, TDFISTANet
from libs.cs import CSConfiguration, OMPCS
from libs.visual import CSPlot, NetPlot, CSNetPlot
from libs.CSI_Frame import COMM_RMNet_CSI, COMM_RMNetStu_CSI, FISTANet_CSI, ISTANet_CSI
from libs.CSI_Frame import CSPNet_CSI, COMM_CSINet_CSI, COMM_CSINetStu_CSI, COMM_CS_CSI, ISTANetPlus_CSI, TD_FISTANet_CSI


class Xun_Lian_He_Ce_Shi(object):


    def net_csi(self):
        comm_rmnet = COMM_RMNet_CSI()
        cspnet = CSPNet_CSI()
        comm_rmnet_stu = COMM_RMNetStu_CSI()
        comm_csinet = COMM_CSINet_CSI()
        td_fista = TD_FISTANet_CSI()
        comm_fista = FISTANet_CSI()
        comm_csinet_stu = COMM_CSINetStu_CSI()
        ista_plus = ISTANetPlus_CSI()
        ista = ISTANet_CSI()

        td_fista.net_train(ratio=16, layer_num=5, epoch=400)
        td_fista.net_test(ratio=16, layer_num=5)

    def cs_csi(self):
        cs = COMM_CS_CSI()
        cs.cs_register(OMPCS, CSConfiguration.sparse_eye, CSConfiguration.omp)
        cs.CS_joint_test()


class Tu_He_Biao(object):

    @criteria_loop_wrapper
    def all_plot(self, criteria, img_format="svg", ratio_list=config.ratio_list):
        for ratio in ratio_list:
            self.one_plot(criteria, ratio, img_format)

    def one_plot(self, criteria, ratio, img_format="svg"):
        """绘制图像"""
        networks = [
            RMNetConfiguration.network_name,
            CSINetConfiguration.network_name,
            RMNetStuConfiguration.network_name,
            TDFISTANetConfiguration.network_name
        ]
        methods = ["{}-{}".format(CSConfiguration.sparse_eye, CSConfiguration.omp)]
        NetPlot().draw("comm", networks, criteria, img_format, ratio)
        CSPlot().draw("comm", methods, criteria, img_format, ratio)
        CSNetPlot().draw("comm", networks, methods, criteria, img_format, ratio)


if __name__ == '__main__':
    pass



    