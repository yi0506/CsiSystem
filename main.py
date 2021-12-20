"""主函数"""
from libs import config
from libs.utils import v_loop_wrapper, criteria_loop_wrapper
from libs.RM_net import RMNetConfiguration
from libs.CSI_net import CSINetConfiguration
from libs.cs import CSConfiguration, SAMPCS, OMPCS, SPCS, DCTCS, FFTCS
from libs.visual import HS_CSPlot, HS_NetPlot, HS_TimeForm
from libs.CSI_Frame import RMNet_CSI, HS_CSINet_CSI, HS_CS_CSI, RMNetStu_CSI


class Xun_Lian_He_Ce_Shi(object):


    def net_csi(self):
        rmnet = RMNet_CSI()
        csinet = HS_CSINet_CSI()

        rmnet.net_joint_train(epoch=100)
        rmnet.net_joint_test()
        csinet.net_joint_train(epoch=70)
        csinet.net_joint_test()

        rm_stu = RMNetStu_CSI()
        rm_stu.net_joint_train(epoch=50)
        rm_stu.net_joint_test()

    def cs_csi(self):
        cs = HS_CS_CSI()
        cs.cs_register(OMPCS, CSConfiguration.sparse_dct, CSConfiguration.omp)
        cs.CS_joint_test()


class Tu_He_Biao(object):

    @criteria_loop_wrapper
    @v_loop_wrapper
    def all_plot(self, v, criteria, img_format="svg", ratio_list=config.ratio_list):
        self.one_plot(v, criteria, img_format, ratio_list)

    def one_plot(self, v, criteria, img_format="svg", ratio_list=config.ratio_list):
        """绘制图像"""
        networks = [
            RMNetConfiguration.network_name,
            CSINetConfiguration.network_name
        ]
        methods = ["{}-{}".format(CSConfiguration.sparse_dct, CSConfiguration.omp)]
        HS_NetPlot().draw(v, networks, criteria, img_format, ratio_list)
        HS_CSPlot().draw(v, methods, criteria, img_format, ratio_list)

    @v_loop_wrapper
    def all_form(self, v, ratio_list=config.ratio_list):
        self.one_form(v, ratio_list)

    def one_form(self, v, ratio_list=config.ratio_list):
        """输出系统耗时json文件，cs、old_csi、net之间的对比"""
        networks = [
            RMNetConfiguration.network_name,
            CSINetConfiguration.network_name
        ]
        methods = ["{}-{}".format(CSConfiguration.sparse_dct, CSConfiguration.omp)]
        tf = HS_TimeForm()
        tf.csi_time_form(v, methods, networks, ratio_list)


if __name__ == '__main__':
    x1 = Xun_Lian_He_Ce_Shi()
    y1 = Tu_He_Biao()
    x1.net_csi()
    # x1.cs_csi()
    # y1.all_form()
    # y1.all_plot()


