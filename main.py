"""主函数"""
from libs import config
from libs.utils import criteria_loop_wrapper
from libs.RM_net import RMNetConfiguration
from libs.CSI_net import CSINetConfiguration
from libs.cs import CSConfiguration, OMPCS
from libs.visual import CSPlot, NetPlot, TimeForm
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

        cspnet.net_train(ratio=16, epoch=400)
        cspnet.net_test(ratio=16)

        comm_fista.net_train(ratio=16, layer_num=5, epoch=400)
        comm_fista.net_test(ratio=16, layer_num=5)
        
        comm_rmnet.net_train(ratio=16, epoch=400)
        comm_rmnet.net_test(ratio=16, )

        comm_rmnet_stu.net_train(ratio=16, epoch=400)
        comm_rmnet_stu.net_test(ratio=16)

        comm_csinet.net_train(ratio=16, epoch=400)
        comm_csinet.net_test(ratio=16)

        comm_csinet_stu.net_train(ratio=16, epoch=400)
        comm_csinet_stu.net_test(ratio=16)

        ista_plus.net_train(ratio=16, layer_num=5, epoch=400)
        ista_plus.net_test(ratio=16, layer_num=5)

        ista.net_train(ratio=16, layer_num=5, epoch=400)
        ista.net_test(ratio=16, layer_num=5)

    def cs_csi(self):
        cs = COMM_CS_CSI()
        cs.cs_register(OMPCS, CSConfiguration.sparse_dct, CSConfiguration.omp)
        cs.CS_joint_test()


class Tu_He_Biao(object):

    @criteria_loop_wrapper
    def all_plot(self, criteria, img_format="svg", ratio_list=config.ratio_list):
        self.one_plot(criteria, img_format, ratio_list)

    def one_plot(self, criteria, img_format="svg", ratio_list=config.ratio_list):
        """绘制图像"""
        networks = [
            RMNetConfiguration.network_name,
            CSINetConfiguration.network_name
        ]
        methods = ["{}-{}".format(CSConfiguration.sparse_dct, CSConfiguration.omp)]
        NetPlot().draw("comm", networks, criteria, img_format, ratio_list)
        CSPlot().draw("comm", methods, criteria, img_format, ratio_list)

    def all_form(self, ratio_list=config.ratio_list):
        self.one_form(ratio_list)

    def one_form(self, v, ratio_list=config.ratio_list):
        """输出系统耗时json文件，cs、old_csi、net之间的对比"""
        networks = [
            RMNetConfiguration.network_name,
            CSINetConfiguration.network_name
        ]
        methods = ["{}-{}".format(CSConfiguration.sparse_dct, CSConfiguration.omp)]
        tf = TimeForm()
        tf.csi_time_form("comm", methods, networks, ratio_list)


if __name__ == '__main__':
    x1 = Xun_Lian_He_Ce_Shi()
    y1 = Tu_He_Biao()
    x1.net_csi()
    x1.cs_csi()
    y1.all_form()
    y1.all_plot()

