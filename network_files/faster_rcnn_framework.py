
from torch import nn
import torch.nn.functional as F
from .roi_head import RoIHeads
from .rpn_function import RegionProposalNetwork,RPNHead,AnchorsGenerator
from torchvision.ops import MultiScaleRoIAlign

#  faster rcnn部分中的两个全连接层
class TwoMLPHead(nn.Module):
    """
    该类是接受经过roialign处理之后的boxfeature，
    其中包含将boxfeature展平，以及两个全连接层，将输出的最后一个维度转化为1024

    Arguments:
        in_channels (int): feature_channel(256)*roialign_size(7)**2 = 256*7*7
        representation_size (int): 1024
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)  # 数据展平处理[b*512,256*7*7]

        x = F.relu(self.fc6(x))    # 改变维度大小 [b*512,1024]
        x = F.relu(self.fc7(x))    # [b*512,1024]

        return x


 #   两个全连接层后的预测层，对类别和边界框回归参数进行预测
class FastRCNNPredictor(nn.Module):
    """
    标准分类 + 边界框回归参数 
  

    Arguments:
        in_channels (int): 输入的通道数，等于TwoMLPHead这个类的返回值的第二个维度的大小(1024)
        num_classes (int): 物体的种类(包含背景)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        # 两个全连接层
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        # 类别的预测 [b*512,num_classes+1]
        scores = self.cls_score(x)
        # 回归参数的预测[b*512,(num_classes+1)*4]
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas 

# faster Rcnn 主干网络搭建
class FasterRCNN(nn.Module):
   
    def __init__(self, backbone, num_classes=None, # 主干网络和预测类别个数

        # RPN 部分的相关参数
        # rpn中在nms处理前保留的proposal数（training,testing）
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, 
        # rpn中在nms处理后保留的proposal数   
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  
        # rpn中进行nms处理时使用的iou阈值
        rpn_nms_thresh=0.7,  
        # rpn计算损失时，采集正负样本设置的阈值
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        # rpn计算损失时采样的样本数，以及正样本占总样本的比例  
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        # rpn中对取出的proposal进行小概率滤除的阈值，之后进行nms处理  
        rpn_score_thresh=0.0, 

        # faster分支参数
        # nms处理之前移除低目标概率         
        box_score_thresh=0.05,
        #fast rcnn中进行nms处理的阈值
        box_nms_thresh=0.5, 
        # 对预测结果根据score排序取前100个目标
        box_detections_per_img=100,

        # fast rcnn计算误差时，采集正负样本设置的阈值
        box_fg_iou_thresh=0.5, # 上限（大于即为正样本）
        box_bg_iou_thresh=0.5,  # 下限（小于即为负样本）

        # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
        box_batch_size_per_image=512, box_positive_fraction=0.25,  
        representation_size = 1024,    # faster分支部分两个全连接层的输出通道数
        bbox_reg_weights=None  # 边界框回归时 xywh所占的比重

        
        ):
        
        super(FasterRCNN, self).__init__()


        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 生成针对resnet50_fpn的anchor生成器
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )
        # 生成RPN通过滑动窗口预测网络部分
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        #  Multi-scale RoIAlign pooling
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
            output_size=[7, 7],
            sampling_ratio=2)

        # fast RCNN中roi pooling后的展平处理两个全连接层部分

        resolution = box_roi_pool.output_size[0]  # 默认等于7
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size
            )

        # 在box_head的输出上预测部分
        box_predictor = FastRCNNPredictor(representation_size,num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起

        self.roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100
