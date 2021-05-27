"""主函数"""
import libs


def main():
    """根据需求，使用不同的方法"""
    # encoder开始不使用fc层？
    csi = libs.MyCsi()
    # csi.model_train()
    # csi.model_test()
    # csi.model_joint_train(epoch=50, multi=False)
    # csi.model_joint_test(multi=False)
    # csi.old_csi_train()
    # csi.old_csi_test()
    # csi.cs_test()
    csi.visual_display(img_format="eps")
    print("~\(≧ w ≦)/~  砸瓦鲁多!!!!!!!!!")


if __name__ == '__main__':
    main()
