import os
import datetime

import torch
from torchvision.ops.misc import FrozenBatchNorm2d

from data_set import data_transforms as transforms
from network_files.mask_rcnn import MaskRCNN
from backbone.resnet_fpn_model import resnet_fpn_backbone
from data_set.my_dataset import CocoDetection
from train_utils import train_eval_utils as utils
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def create_model(num_classes, parameter):
    # 如果GPU显存很小，batch_size不能设置很大，建议将norm_layer设置成FrozenBatchNorm2d(默认是nn.BatchNorm2d)

    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d,
    #                                  trainable_layers=3)

    backbone = resnet_fpn_backbone(pretrain_path="", trainable_layers=5,Framework = parameter.framework,Is_se=parameter.SE) #pre_training_weights\\resnet50.pth

    model = MaskRCNN(backbone,num_classes=num_classes,parameters = parameter)

    if args.pretrain:   # bool 是否导入预训练权重
        
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        # pre_training_weights\\maskrcnn_resnet50_fpn_coco.pth
        weights_dict = torch.load("pre_training_weights\\maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    return model


def main(args):
    
    # 获取训练设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # now返回当前时刻的时间信息
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"result_map_loss\\det_results{now}.txt"    # 目标检测的验证结果
    seg_results_file = f"result_map_loss\\seg_results{now}.txt"    # 实例分割的验证结果

    # 由于采集图像时图像会水平翻转，因此在train和val时需翻转回去
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(1)]),
        "val": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(1)])
    }
    
    # 一次性输入网络的训练的图片数量
    batch_size = args.batch_size
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # 训练数据集的根目录data\\coco
    data_root = args.data_path
    # 导入训练数据
    train_dataset = CocoDetection(data_root, "train", data_transform["train"])

    # 提取数据文件 loader
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)

    # 导入验证集文件
    val_dataset = CocoDetection(data_root, "val", data_transform["val"])

    # 将文件打包成batch
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    # 创建网络结构 输入物体的种类； 训练时是否导入预训练权重
    model = create_model(num_classes=args.num_classes + 1, parameter = args)
    # 设置训练的设备（默认GPU）
    model.to(device)

    # 绘图信息
    train_loss = []   #平均总损失
    learning_rate = []  # 学习率
    val_map = []      #验证map值
    accuracy = []


    # 模型中的权重与偏差值等信息
    params = [p for p in model.parameters() if p.requires_grad]

    # 优化器参数
    # params神经网络中的权重、偏置信息；lr学习率
    # momentum冲量
    # weight_decay权重衰减防止过拟合
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 实例化混合精度对象
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 动态调整学习率
    # lr_steps为第几次训练修改学习率
    # lr_gamma为学习率的修改倍数
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)
    
    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:

        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

# ***************************************************************
    # 开始训练次数与总训练次数
    for epoch in range(args.start_epoch, args.epochs):
        
        # 训练函数
        # model 网络模型
        # optimizer 设置的优化器
        # train_data_loader  导入训练数据集
        # device 训练的设备（默认GPU）
        # epoch 当前的训练次数
        # 返回值：平均损失；学习率
        LOSS = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=100,
                                              warmup=True, scaler=scaler)
        train_loss.append([LOSS[0],LOSS[1]])
        
        
        # learning_rate.append(lr)

        # 更新学习率
        lr_scheduler.step()

        ###### 使用验证集评估
        ACCURACY = utils.evaluate(model, val_data_loader, args,device=device)
        accuracy.append(ACCURACY)
        # 写入定义好的目标检测文件
        with open(det_results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = LOSS+ACCURACY
            result_info = [str(x) for x in result_info]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # 写入定义好的实例分割文件
        # with open(seg_results_file, "a") as f:
        #     # 写入的数据包括coco指标还有loss和learning rate
        #     result_info = [round(i,4) for i in seg_info ]+[round(mean_loss.item(),4)] + LOSS+ACCURACY
        #     result_info = [str(x) for x in result_info]
        #     txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        #     f.write(txt + "\n")

       

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "./save_weights/model_{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 :
        from plot_data.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, accuracy)

    # plot mAP curve
    # if len(val_map) != 0:
    #     from plot_data.plot_curve import plot_map
    #     plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='data\\mass', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=3, type=int, help='num_classes')
    # 权重文件保存地址
    parser.add_argument('--output-dir', default='.\save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[10,15,20,30], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.02, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练的batch size(如果内存/GPU显存充裕，建议设置更大)
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=False, help="load COCO pretrain weights.")
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")


    ################################# mass prediction 和mass score设置参数
    parser.add_argument('--score-begin-epoch', default=0, type=int,help='从第几个epoch开始计算score损失')
    parser.add_argument('--mass-number', default=4, type=int,help='预测mass的数量')
    parser.add_argument('--senor', default=30, type=int,help='传感器的最大量程')
    parser.add_argument('--mass-label', default=[2,3], nargs='+', type=int,
                        help='石子标签2  物料标签3')
    parser.add_argument('--truck-label', default=[1], nargs='+', type=int,
                        help='车辆标签1')
    parser.add_argument('--initial-para', default=[0,0,0,0], nargs='+', type=float,
                        help='未放置物料时，传感器的初始值[6.9632,6.9011,4.0415,3.9160]')
    parser.add_argument('--cat-style', default='Conv', 
                        help='score的拼接类型：Conv 、Line和No ，其中No表示不拼接。注意只有mass-number等于4才可以用Conv')
    parser.add_argument('--SE', default=False, 
                        help='backbone中是否启用se')
    parser.add_argument('--framework', default=101,  type=int,
                        help='34 50 101')


    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)
