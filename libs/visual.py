# -*- coding: UTF-8 -*-
"""对CSI反馈系统测试结果进行可视化处理"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import json

import config
from utils import rec_mkdir


class LoadTestData(object):
    """加载测试数据"""

    def load_net_data(self, v: int, ratio: int, networks: list, model_snr: int) -> dict:
        """
        加载不同网络的测试结果，
        Parameter:
            networks: 所有网络的集合
            
        Return:
            y_dict
            
            y_dict: {
                "RMNet": {
                            "0dB":{loss、similarity、time、capacity},
                            "2dB": {loss、similarity、time、capacity}
                        }, 
                "CSINet": {...}
                }

        """

        y_dict = dict()
        if not networks:
            return y_dict
        for nw in networks:
            y_dict[nw] = pickle.load(open("./test_result/{}km/{}/ratio_{}/{}_{}dB.pkl".format(v, nw, ratio, nw, model_snr), "rb"))
        return y_dict

    def load_cs_data(self, v, ratio, methods):
        """
        加载cs的测试结果
        
        methods: ["dct-omp", "dct-sp",...]
        
        y_dict：不同压缩率下，每种方式在不同信噪比下的测试结果,包含不同压缩率的y_one_ratio，
        y_dict: {
                    "dct_idct": {"0dB":{similarity, time, loss, capacity}, "5dB":{},...},
                    "dct_omp":{...},
                }
        """
        y_dict = dict()  # 保存全部压缩率的结果
        if not methods:
            return y_dict
        for md in methods:
            y_dict[md] = pickle.load(open("./test_result/{}km/cs/ratio_{}/result/{}.pkl".format(v, ratio, md), "rb"))
        return y_dict


class Plot(object):
    """
    对测试的结果进行可视化展示，并保存图片
    
        description: 图片描述
        description = {
            "xlable": x轴描述，
            "ylable": y轴描述，
            "title": 图片标题，
            "loc": 图例位置,
            "use_gird": 是否添加网格线,
            "img_name": 图片名字
        }
        model_snr: 取哪个信噪比模型的测试数据进行绘图
        best_model: 是否选择最佳模型的结果，不同信噪比分别使用相应信噪比下训练好的模型的测试数据进行绘图
        velocity: 移动速度
    
    
    """
    marker_list = ["o", "*", "d", "v", "s", "x", "h", "^", "<"]
    line_style = ["solid", "dashed", "dashdot", "dotted"]
    FRONT = {
        "font.family": 'serif',
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
        "axes.unicode_minus": False,
    }
    rcParams.update(FRONT)
    dl = LoadTestData()

    @staticmethod
    def trans_criteria(criteria):
        """
        对字符串进行转化：
                a.将"time"转化为 "Time"
                b.将"相似度"转化为 "\rho"
                c.其余不变

        :return: "Time", "\rho", "NMSE", "Capacity"
        """
        if criteria == "time":
            return "Time"  # time大写
        elif criteria == "相似度":
            return r"\rho"  # 将"相似度"转化为希腊字母"rho"
        else:
            return criteria
        
    @staticmethod
    def set_description(title, xlabel, ylabel, img_name, **description):
        """根据描述信息画图"""
        # 非必须
        use_grid = description.get("use_grid", False)
        xticks = description.get("xticks")
        yticks = description.get("yticks")
        loc = description.get("loc", "best")
        yscale = description.get("yscale")
        # 添加描述信息
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=15)
        plt.legend(loc=loc, prop={'family': "Times New Roman", "size": 12})
        if yscale:
            plt.yscale('log')
        if yticks:
            plt.yticks(ticks=yticks, fontproperties="Times New Roman")
        else:
            plt.yticks(fontproperties="Times New Roman")
        if xticks:
            plt.xticks(ticks=xticks, fontproperties="Times New Roman")
        else:
            plt.xticks(fontproperties="Times New Roman")
        if use_grid:
            plt.grid(linestyle="--", alpha=0.4)
        rec_mkdir(img_name)
        plt.savefig(img_name, bbox_inches='tight', pad_inches=0)
        plt.close()
        print("{} was done.".format(img_name))


class HighSpeedNetPlot(Plot):

    def net_plot(self, v, networks, criteria, model_snr, img_format, ratio_list):
        """
        :param ratio_list: 压缩率列表
        :param img_format: 图片格式
        :param model_snr: 不同信噪比下训练的模型
        :param v: 速度
        :param networks: 不同网络模型
        :param criteria: "loss"、"相似度"、"time"、"capacity"
        比较不同压缩率的情况下，不同网络的随着snr的变化的效果
        """
        x = config.SNRs
        plt.figure()
        for ratio in ratio_list:
            # 获取数据
            data = self.dl.load_net_data(v, ratio, networks, model_snr)
            # 解析数据
            for idx, (network, val) in enumerate(data.items()):
                y = []
                for dB in x:
                    y.append(val["{}dB".format(dB)][criteria])
                plt.plot(x, y, label=r"1/{} {}".format(ratio, network), marker=self.marker_list[idx], linestyle=self.line_style[idx], markerfacecolor="none", markersize=8, linewidth=2)
        # 添加描述信息
        criteria_label = self.trans_criteria(criteria)
        title = r"$\mathrm{{{}km/h}}$".format(v)
        xlabel = r"$\mathrm{SNR(dB)}$"
        ylabel = r"$\mathrm{{{}\/\/(bps/Hz)}}$".format(criteria_label) if criteria == "Capacity" else r"$\mathrm{{{}}}$".format(criteria_label)
        img_name = "./images/{0}km/net-{1}—{2}dB_model.{3}".format(v, criteria, model_snr, img_format)
        use_gird = True
        yticks = config.y_ticks_similarity if criteria == "相似度" else None
        loc = "best"  # "lower right"
        self.set_description(title, xlabel, ylabel, img_name, use_gird=use_gird, yticks=yticks, loc=loc)


class CSPlot(Plot):

    def cs_plot(self, v, methods, criteria, img_format, ratio_list):
        """
        :param ratio_list: 压缩率列表
        :param img_format: 图片格式
        :param methods: 不同CS方法
        :param v: 速度
        :param criteria: "loss"、"similarity"、"time"、"capacity"
        在不同压缩率下，绘制不同cs方法的criteria随着snr的变化
        """
        x = config.SNRs
        plt.figure()
        for ratio in ratio_list:
            # 获取数据
            data = self.dl.load_cs_data(v, ratio, methods)
            # 解析数据
            for idx, (methdod, val) in enumerate(data.items()):
                y = []
                for dB in x:
                    y.append(val["{}dB".format(dB)][criteria])
                plt.plot(x, y, label=r"1/{} {}".format(ratio, methdod), marker=self.marker_list[idx], linestyle=self.line_style[idx], markerfacecolor="none", markersize=8, linewidth=2)
        # 添加描述信息
        criteria_label = self.trans_criteria(criteria)
        title = r"$\mathrm{{{}km/h}}$".format(v)
        xlabel = r"$\mathrm{SNR(dB)}$"
        ylabel = r"$\mathrm{{{}\/\/(bps/Hz)}}$".format(criteria_label) if criteria == "Capacity" else r"$\mathrm{{{}}}$".format(criteria_label)
        img_name = "./images/{0}km/CS-{1}.{2}".format(v, criteria, img_format)
        use_gird = True
        yticks = config.y_ticks_similarity if criteria == "相似度" else None
        loc = "best"  # "lower right"
        self.set_description(title, xlabel, ylabel, img_name, use_gird=use_gird, yticks=yticks, loc=loc)


class CSNetMultiRatio(Plot):

    def cs_plot_multi_ratio(self, v, networks, model_snr, methods, criteria, img_format, ratio_list):
        """
        :param model_snr: 不同信噪比下训练的模型
        :param networks: 不同网络模型
        :param ratio_list: 压缩率列表
        :param img_format: 图片格式
        :param methods: 不同CS方法
        :param v: 速度
        :param criteria: "loss"、"相似度"、"time"、"capacity"
        在的不同缩率下，绘制不同cs方法的criteria随着snr的变化，同时与神经网络进行对比
        """
        x = config.SNRs
        plt.figure()
        for ratio in ratio_list:
            # 获取CS数据
            data_cs = self.dl.load_cs_data(v, ratio, methods)
            # 解析数据
            for methdod, val in data_cs.items():
                y = []
                for dB in x:
                    y.append(val["{}dB".format(dB)][criteria])
                plt.plot(x, y, label=r"1/{} {}".format(ratio, methdod), marker=self.marker_list[0], linestyle=self.line_style[0], markerfacecolor="none", markersize=8, linewidth=2)
            # 获取神经网络数据
            data_net = self.dl.load_net_data(v, ratio, networks, model_snr)
            # 解析数据
            for network, val in data_net.items():
                y = []
                for dB in x:
                    y.append(val["{}dB".format(dB)][criteria])
                plt.plot(x, y, label=r"1/{} {}".format(ratio, network), marker=self.marker_list[1], linestyle=self.line_style[1], markerfacecolor="none", markersize=8, linewidth=2)

        # 添加描述信息
        criteria_label = self.trans_criteria(criteria)
        title = r"$\mathrm{{{}km/h}}$".format(v)
        xlabel = r"$\mathrm{SNR(dB)}$"
        ylabel = r"$\mathrm{{{}\/\/(bps/Hz)}}$".format(criteria_label) if criteria == "Capacity" else r"$\mathrm{{{}}}$".format(criteria_label)
        img_name = "./images/{0}km/CS-Net-{1}-{2}dB_model-{3}.{4}".format(v, criteria, model_snr, ratio_list, img_format)
        use_gird = True
        yticks = config.y_ticks_similarity if criteria == "相似度" else None
        loc = "best"  # "lower right"
        self.set_description(title, xlabel, ylabel, img_name, use_gird=use_gird, yticks=yticks, loc=loc)


class TimeForm(Plot):
    """生成系统耗时的Json数据文件"""

    def csi_time_form(self, v, methods, networks, ratio_list, model_snr):
        """在不同压缩率下，对不同算法的系统耗时以Json形式输出"""
        x = config.SNRs
        time_dic = {}
        for ratio in ratio_list:
            time_dic[ratio] = {}
            # 神经网络数据
            net_data = self.dl.load_net_data(v, ratio, networks, model_snr)
            for network, val in net_data.items():
                # 取平均时间
                y = 0
                for dB in x:
                    y += val["{}dB".format(dB)]["time"]
                time_dic[ratio][network] = y / len(x)
            # CS数据
            cs_data = self.dl.load_cs_data(v, ratio, methods)
            for method, val in cs_data.items():
                y = 0
                for dB in x:
                    y += val["{}dB".format(dB)]["time"]
                time_dic[ratio][method] = y / len(x)
        file_path = "./images/{0}km/time.json".format(v)
        rec_mkdir(file_path)
        json.dump(time_dic, open(file_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        print("{} is done".format(file_path))


if __name__ == '__main__':
    pass
