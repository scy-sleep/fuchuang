import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\ppsoftware\yolo11\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml')
    model.train(data=r'D:\ppsoftware\yolo11\ultralytics-main\datasets\hot\data.yaml',
                cache=False,
                imgsz=640,
                epochs=10,  # 分几片
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )

    # 训练完成后
    print("!!!!!!!!!!!!!!!!!训练完毕!!!!!!!!!!!!!!!!!!!!!!!")
    results = model.val(
        data=r'D:\ppsoftware\yolo11\ultralytics-main\datasets\hot\data.yaml',  # 确保 data.yaml 中有 test 路径
        split='test',  # 指定使用测试集
        batch=16,
        device='0',
        plots=True,  # 生成混淆矩阵、PR 曲线等图表
        save_json=True,  # 保存评估结果为 JSON 文件
        save_hybrid=True  # 保存混合标签和预测结果的可视化
    )
    print("加载完了应该！！！！！！！！！！！！！！！！！")


    # 输出关键指标
    print(f"mAP@50: {results.box.map50:.4f}")  # IoU=0.5 时的 mAP
    print(f"mAP@50-95: {results.box.map:.4f}")  # IoU=0.5:0.95 的平均 mAP
    print(f"Precision: {results.box.mp:.4f}")  # 精确率
    print(f"Recall: {results.box.mr:.4f}")  # 召回率



    # # 获取 mAP@50 (IoU=0.5 时的 mAP)
    # print(f"mAP@50: {results.box.map50}")
    #
    # # 获取 mAP@50-95 (IoU=0.5:0.95 的平均 mAP)
    # print(f"mAP@50-95: {results.box.map}")

    # print(results.mAP)  # 输出测试集上的 mAP 指标


