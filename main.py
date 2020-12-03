"""主函数"""
import libs


def main():
    """根据需求，使用不同的方法"""
    csi = libs.MyCsi(v_list=[50, 100, 150, 300])
    # csi.model_train(400, None)
    # csi.model_test(None)
    # csi.model_joint_train(epoch=200)
    # csi.model_joint_test([None])
    # csi.old_csi_train()
    # csi.old_csi_test()
    # csi.cs_test()
    csi.visual_display()
    print("~\(≧ w ≦)/~  砸瓦鲁多!!!!!!!!!")


if __name__ == '__main__':
    main()
