_base_ = 'yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# fast means faster training speed,
# but less flexibility for multitasking
model = dict(
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True))



# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs 对应lr=0.01
base_lr = 0.01
max_epochs = 150  # Maximum training epochs


# 数据集类别名称
class_name = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
num_classes = len(class_name)  # Number of classes for classification
# metainfo 必须要传给后面的 dataloader 配置，否则无效
# palette 是可视化时候对应类别的显示颜色
# palette 长度必须大于或等于 classes 长度
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                                             (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                                             (0, 0, 192), (250, 170, 30)])

# 加载 COCO 预训练权重
load_from = 'demo/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'

train_dataloader = dict(collate_fn=dict(type='yolov5_collate'),
                        dataset=dict(
                            metainfo=metainfo)
                        )

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo))

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
