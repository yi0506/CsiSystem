# -*- coding: UTF-8 -*-
"""对CSI反馈系统测试结果进行可视化处理"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import os
import json

from libs.train import rec_mkdir
from libs import config


PREFERENCE = {
        "font.family": 'serif',
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
        "axes.unicode_minus": False
}
rcParams.update(PREFERENCE)


class LoadTestData(object):
    """加载测试数据"""

    def __init__(self, **kwargs):
        """
        Parameter:

            self.dtype - 加载的数据类型，net为神经网络模型测试数据，cs为压缩感知测试数据
            self.velocity - 选择的速度
            self.ratio_list - 选择的压缩率列表

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, **kwargs):
        """根据data_type加载不同的测试数据并返回"""
        if self.dtype == "net":
            return self.load_net_data(**kwargs)
        elif self.dtype == "cs":
            return self.load_cs_data(**kwargs)
        else:
            return exit("data_type输入错误")

    def load_net_data(self, **kwargs):
        """
        加载网络的测试结果，
        y_dict为不同压缩率的数据
        y_dict: {"seq2seq": {}, "old_csi": {}}

        y_dict["seq2seq"]: {
                            "nonedB":{"2":{"0dB":{loss、similarity、time},
                            "5dB":{},...}, "4":{},...}, "5dB":{}
                        }

        y_dict["old_csi"]: {
                            "2": {"-10dB":{loss、similarity、time}, "20dB":{}, ...},
                            "4": {"10dB":{loss、similarity、time},{}, .....
                        }
        """
        model_snr = kwargs.get('model_snr')
        y_dict = dict()
        net_dict = dict()
        old_csi_dict = dict()
        # 读取csi_net测试结果
        for snr in model_snr:
            temp_csi = dict()
            for rate in self.ratio_list:
                data = pickle.load(open("./test_result/{}km/csi_net/csi_net_{}_{}dB.pkl".format(self.velocity, rate, snr), "rb"))
                temp_csi[str(rate)] = data
            net_dict["{}dB".format(snr)] = temp_csi
        # 读取old_csi_net测试结果
        for rate in self.ratio_list:
            old_csi_data = pickle.load(open("./test_result/{}km/old_csi/old_csi_{}.pkl".format(self.velocity, rate), "rb"))
            old_csi_dict[str(rate)] = old_csi_data
        y_dict["seq2seq"] = net_dict
        y_dict["old_csi"] = old_csi_dict
        return y_dict

    def load_cs_data(self, **kwargs):
        """
        加载cs的测试结果，
        y_dict：不同压缩率下，每种方式在不同信噪比下的测试结果,包含不同压缩率的y_one_ratio，
        y_dict: {"2": y_one_ratio, "4": y_one_ratio, .....,}
        y_one_ratio: {
                        "dct_idct": {"0dB":{similarity, time, loss}, "5dB":{},...},
                         "dct_dct_omp":{}, ....,
                        "method":["dct_idct", "dct_omp",...],
                        "net":{"0dB":{loss、similarity、time}, "5dB":{}, ...}
                        "old_csi":{"0dB":{loss、similarity、time}, "5dB":{}, ...}
                        "snr_model": "{}dB".format(snr_model)
                    }
        """
        model_snr = kwargs.get('model_snr')
        y_dict = dict()  # 保存全部压缩率的结果
        for Fi_ratio in self.ratio_list:
            dir_path = "./test_result/{}km/cs/ratio_{}/result/".format(self.velocity, Fi_ratio)
            file_list = os.listdir(dir_path)
            method_list = list()
            y_one_ratio = dict()  # 保存一个压缩率下的结果
            # 加载cs测试结果
            for file_name in file_list:
                file_path = dir_path + file_name
                if os.path.isfile(file_path):
                    data = pickle.load(open(file_path, "rb"))  # 一种CS方法的仿真结果
                    y_one_ratio[file_name[:-6].upper()] = data
                    method_list.append(file_name[:-6].upper())
            y_one_ratio["method"] = method_list
            # 加载csi_net测试结果
            net_data = pickle.load(open("./test_result/{}km/csi_net/csi_net_{}_{}dB.pkl".format(self.velocity, Fi_ratio, model_snr), "rb"))
            # 加载old_csi_net测试结果
            old_csi_data = pickle.load(open("./test_result/{}km/old_csi/old_csi_{}.pkl".format(self.velocity, Fi_ratio), "rb"))
            y_one_ratio["net"] = net_data
            y_one_ratio["old_csi"] = old_csi_data
            y_dict[str(Fi_ratio)] = y_one_ratio
        return y_dict


class Plot(object):
    """对测试的结果进行可视化展示，并保存图片"""
    marker_list = ["o", "*", "d", "v", "s", "x", "h", "^", "<"]
    ratio_list = config.ratio_list
    snr_list = config.SNRs
    line_style = ["solid", "dashed", "dashdot", "dotted"]
    y_ticks_similarity = config.y_ticks_similarity

    def __init__(self, data=None):
        """
        :param data: 由 LoadTestData().load_data()的返回值决定，data_type不同，data的形式不同

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
        self.data = data

    def net_plot(self, criteria, **kwargs):
        """
        :param criteria: "loss"、"相似度"、"time"
        不同压缩率下，criteria随着snr的变化
        纵坐标信噪比，横坐标loss、similarity、time，比较不同压缩率的情况
        对比csi_net与old_csi在不同指标下随着snr的变化
        """
        model_snr = kwargs.get("model_snr", None)
        best_model = kwargs.get("best_model", False)
        velocity = kwargs.get("velocity", config.velocity)
        loc = kwargs.get("loc", "best")
        img_format = kwargs.get("img_format", "svg")
        plt.figure()
        num = 0
        for ratio in self.ratio_list:
            y, y_old = self.net_plt_get_net_data(self.data, criteria, best_model, model_snr, ratio)
            # 神经网络在不同压缩率下绘图
            plt.plot(self.snr_list, y, label=r"1/{} RM-Net".format(self.ratio_list[num]), marker=self.marker_list[num], markerfacecolor="none", markersize=8, linewidth=2)
            plt.plot(self.snr_list, y_old, linestyle=self.line_style[1], label=r"1/{} CsiNet".format(self.ratio_list[num]), marker=self.marker_list[num], markerfacecolor="none", markersize=8, linewidth=2)
            num += 1
        img_criteria = self.trans_criteria(criteria)
        description = {
            "xlable": r"$\mathrm{SNR(dB)}$",
            "ylable": r"$\mathrm{{{}}}$".format(img_criteria),
            "title": "$\mathrm{{{}km/h}}$".format(velocity),
            "loc": loc,
            "use_gird": False,
            "img_name": "./images/{0}km/{0}-net-{1}—{2}dB.{3}".format(velocity, criteria, model_snr, img_format) if not best_model else "./images/{}km/{}-net-{}—best.{}".format(velocity, velocity, criteria, img_format),
            "criteria": img_criteria,
        }
        self.plt_description(plt, **description)

    def cs_plot(self, criteria, **kwargs):
        """
        :param criteria: "loss"、"similarity"、"time"
        在相同压缩率下，绘制不同cs方法的criteria随着snr的变化，同时对比csi_net与old_csi
        """
        ratio = kwargs.get("ratio")
        model_snr = kwargs.get("model_snr", None)
        velocity = kwargs.get("velocity", config.velocity)
        loc = kwargs.get("loc", "best")
        img_format = kwargs.get("img_format", "svg")
        plt.figure()
        num = 0
        # 传统cs方法绘图
        for method in self.data[str(ratio)]["method"]:
            y = self.cs_plt_get_cs_data(self.data[str(ratio)][method], criteria)
            plt.plot(self.snr_list, y, label=r"{}".format(method.replace("_", "-")), marker=self.marker_list[num], markerfacecolor="none", markersize=8, linewidth=2)
            num += 1
        # 神经网络方法绘图：csi_net与old_csi_net
        y_csi, y_old_csi = self.cs_plt_get_net_data(self.data[str(ratio)], criteria)
        plt.plot(self.snr_list, y_csi, label=r"RM-Net", marker=self.marker_list[-1], markerfacecolor="none", markersize=8, linewidth=2)
        plt.plot(self.snr_list, y_old_csi, label=r"CsiNet", marker=self.marker_list[-2], markerfacecolor="none", markersize=8, linewidth=2)
        img_criteria = self.trans_criteria(criteria)
        description = {
            "xlable": r"$\mathrm{SNR(dB)}$",
            "ylable": r"$\mathrm{{{}}}$".format(img_criteria),
            "title": "$\mathrm{{{}km/h}}$".format(velocity),
            "loc": loc,
            "use_gird": False,
            "img_name": "./images/{0}km/{0}-cs-snr-{1}-{2}-{3}dB.{4}".format(velocity, criteria, ratio, model_snr, img_format),
            "criteria": img_criteria,
        }
        self.plt_description(plt, **description)

    def cs_plot_multi_ratio(self, criteria, **kwargs):
        """
        :param criteria: "loss"、"相似度"、"time"
        :param kwargs: {
                            ratio_list: 压缩率列表，
                            use_grid: 是否使用网格线，
                            velocity：速度值
                    }
        在的不同缩率下，绘制不同cs方法的criteria随着snr的变化，同时对比csi_net与old_csi
        :return:
        """
        ratio_list = kwargs.get("ratio_list")
        ratio_list = ratio_list if ratio_list is not None else config.ratio_list
        velocity = kwargs.get("velocity", config.velocity)
        used_model = kwargs.get('cs_multi_ratio_model', None)
        loc = kwargs.get("loc", "best")
        img_format = kwargs.get("img_format", "svg")
        plt.figure()
        for idx, ratio in enumerate(ratio_list):
            num = 0
            # 传统cs方法绘图
            for method in self.data[str(ratio)]["method"]:
                y = self.cs_plt_get_cs_data(self.data[str(ratio)][method], criteria)
                plt.plot(self.snr_list, y, linestyle=self.line_style[idx], label=r"1/{} {}".format(ratio, method.replace("_", "-")), marker=self.marker_list[num], markerfacecolor="none", markersize=8, linewidth=2)
                num += 1
            # 神经网络方法绘图：csi_net与old_csi_net
            y_csi, y_old_csi = self.cs_plt_get_net_data(self.data[str(ratio)], criteria)
            plt.plot(self.snr_list, y_csi, linestyle=self.line_style[idx], label=r"1/{} RM-Net".format(ratio), marker=self.marker_list[-1], markerfacecolor="none", markersize=8, linewidth=2)
            plt.plot(self.snr_list, y_old_csi, linestyle=self.line_style[idx], label=r"1/{} CsiNet".format(ratio), marker=self.marker_list[-2], markerfacecolor="none", markersize=8, linewidth=2)
        img_criteria = self.trans_criteria(criteria)
        description = {
                "xlable": r"$\mathrm{SNR(dB)}$",
                "ylable": r"$\mathrm{{{}}}$".format(img_criteria),
                "title": "$\mathrm{{{}km/h}}$".format(velocity),
                "loc": loc,
                "use_gird": False,
                "img_name": "./images/{0}km/{0}-cs-snr-{1}-{2}-{3}dB模型.{4}".format(velocity, criteria, ratio_list, used_model, img_format),
                "criteria": img_criteria,
        }
        self.plt_description(plt, **description)

    def cs_plt_get_net_data(self, data, criteria):
        """cs画图，获取net数据"""
        y_csi = list()
        y_old_csi = list()
        for snr in self.snr_list:
            net_data = data["net"]["{}dB".format(snr)][criteria]  # csi_net测试数据结果
            old_net_data = data["old_csi"]["{}dB".format(snr)][criteria]  # old_csi_net测试数据结果
            y_csi.append(net_data)
            y_old_csi.append(old_net_data)
        return y_csi, y_old_csi

    def cs_plt_get_cs_data(self, data, criteria):
        """cs画图，获取cs数据"""
        y = list()
        for snr in self.snr_list:
            value = data["{}dB".format(snr)][criteria]
            y.append(value)
        return y

    def net_plt_get_net_data(self, data, criteria, best_model, model_snr, ratio):
        """net画图，获取net数据"""
        y = list()
        y_old = list()
        for snr in self.snr_list:
            if best_model is True:
                model_snr = snr  # 使用各自信噪比下的模型
            y.append(data["seq2seq"]["{}dB".format(model_snr)][str(ratio)]["{}dB".format(snr)][criteria])
            y_old.append(data["old_csi"][str(ratio)]["{}dB".format(snr)][criteria])
        return y, y_old

    def plt_description(self, fig, **kwargs):
        """绘制图片描述信息"""
        img_name = kwargs.get("img_name")
        use_grid = kwargs.get("use_grid", False)
        title = kwargs.get("title")
        xlable = kwargs.get("xlable")
        ylable = kwargs.get("ylable")
        legend_loc = kwargs.get("loc")
        criteria = kwargs.get('criteria')
        fig.xlabel(xlable, fontsize=12)
        fig.ylabel(ylable, fontsize=12)
        fig.title(title, fontsize=15)
        if criteria == "NMSE":
            fig.yscale('log')
        if criteria == r"\rho":
            fig.yticks(ticks=self.y_ticks_similarity, fontproperties="Times New Roman")
        else:
            fig.yticks(fontproperties="Times New Roman")
        fig.xticks(fontproperties="Times New Roman")
        fig.yticks(fontproperties="Times New Roman")
        fig.legend(loc=legend_loc, prop={'family': "Times New Roman", "size": 12})
        if use_grid:
            fig.grid(linestyle="--", alpha=0.4)
        rec_mkdir(img_name)
        fig.savefig(img_name, bbox_inches='tight', pad_inches=0)
        fig.close()
        print("{} was done.".format(img_name))

    @staticmethod
    def trans_criteria(criteria):
        """
        对字符串进行转化：
                a.将"time"转化为"Time"
                b.将"相似度"转化为r"\rho"

        :return: "Time", "\rho", "NMSE"
        """
        if criteria == "time":
            return "Time"  # time大写
        elif criteria == "相似度":
            return r"\rho"  # 将"相似度"转化为希腊字母"rho"
        else:
            return criteria


class TimeForm(object, metaclass=type):
    """生成系统耗时的Json数据文件"""
    def __init__(self, data, velocity):
        self.data = data
        self.velocity = velocity

    def csi_time_form(self):
        """在不同压缩率下，对不同算法的系统耗时以表格形式输出,对不同算法取系统耗时最短的作为最终结果"""
        csi_dict = dict()  # 保存一种算法不同压缩率下的系统耗时
        for ratio in config.ratio_list:
            time_dict = dict()  # 保存一种算法同一种压缩率下的系统耗时
            # 对CS算法获取系统耗时
            for method in self.data[str(ratio)]["method"]:
                cs_list = list()  # cs算法系统耗时
                for snr in config.SNRs:
                    cs_time = self.data[str(ratio)][method]["{}dB".format(snr)]["time"]
                    cs_list.append(cs_time)
                time_dict[method] = min(cs_list)  # 对于不同信噪比下的系统耗时，取最小值作为结果
            # 对RM-Net与CsiNet获取系统耗时
            net_list = list()  # RM-Net系统耗时
            old_csi_list = list()  # csi-net系统耗时
            for snr in config.SNRs:
                net_time = self.data[str(ratio)]["net"]["{}dB".format(snr)]["time"]
                csi_time = self.data[str(ratio)]["old_csi"]["{}dB".format(snr)]["time"]
                net_list.append(net_time)
                old_csi_list.append(csi_time)
            time_dict["old_csi"] = min(old_csi_list)
            time_dict["net"] = min(net_list)
            csi_dict[str(ratio)] = time_dict
        file_path = "./images/{0}km/{0}-time.json".format(self.velocity)
        rec_mkdir(file_path)
        json.dump(csi_dict, open(file_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        print("{} time.json is done".format(self.velocity))


def main_plot(train_models, velocity, ratio_list, cs_multi_ratio_list, img_format, cs_multi_ratio_model):
    """绘制图像"""
    my_plot = Plot()

    # 神经网络：对每种信噪比下的模型单独进行绘图
    net_data_generator = LoadTestData(dtype="net", velocity=velocity, ratio_list=ratio_list)
    my_plot.data = net_data_generator(model_snr=train_models)
    for model_snr in train_models:
        my_plot.net_plot("NMSE", model_snr=model_snr, velocity=velocity, img_format=img_format)
        my_plot.net_plot("相似度", model_snr=model_snr, loc="lower right", velocity=velocity, img_format=img_format)
        my_plot.net_plot("time", model_snr=model_snr, velocity=velocity, img_format=img_format)

    # 神经网络：对于每种信噪比下的结果，选取对应信噪的模型进行绘图
    my_plot.data = net_data_generator(model_snr=config.train_model_SNRs)
    my_plot.net_plot("NMSE", best_model=True, velocity=velocity, img_format=img_format)
    my_plot.net_plot("相似度", best_model=True, loc="lower right", velocity=velocity, img_format=img_format)
    my_plot.net_plot("time", best_model=True, velocity=velocity, img_format=img_format)

    # 传统CS算法绘图，单压缩率
    cs_data_generator = LoadTestData(dtype="cs", velocity=velocity, ratio_list=ratio_list)
    for model_snr in train_models:
        cs_data = cs_data_generator(model_snr=model_snr)
        my_plot.data = cs_data
        for ratio in config.ratio_list:
            my_plot.cs_plot("NMSE", ratio=ratio, model_snr=model_snr, velocity=velocity, img_format=img_format)
            my_plot.cs_plot("相似度", ratio=ratio, model_snr=model_snr, velocity=velocity, img_format=img_format)
            my_plot.cs_plot("time", ratio=ratio, model_snr=model_snr, velocity=velocity, img_format=img_format)

    # 传统CS算法绘图，多压缩率合并，与选择的网络模型进行比较
    cs_data = cs_data_generator(model_snr=cs_multi_ratio_model)
    my_plot.data = cs_data
    my_plot.cs_plot_multi_ratio("NMSE", ratio_list=cs_multi_ratio_list, velocity=velocity, model=cs_multi_ratio_model, img_format=img_format)
    my_plot.cs_plot_multi_ratio("相似度", ratio_list=cs_multi_ratio_list, velocity=velocity, model=cs_multi_ratio_model, img_format=img_format)
    my_plot.cs_plot_multi_ratio("time", ratio_list=cs_multi_ratio_list, velocity=velocity, model=cs_multi_ratio_model, img_format=img_format)


def main_form(velocity, ratio_list, model):
    """输出系统耗时json文件，cs、old_csi、net之间的对比"""
    cs_data = LoadTestData(dtype='cs', velocity=velocity, ratio_list=ratio_list)(model_snr=model)
    form = TimeForm(data=cs_data, velocity=velocity)
    form.csi_time_form()


if __name__ == '__main__':
    pass
