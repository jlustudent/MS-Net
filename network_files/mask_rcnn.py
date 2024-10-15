from collections import OrderedDict
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign
from.faster_rcnn_framework import FasterRCNN
from.transform import GeneralizedRCNNTransform
from typing import Tuple, List, Dict
from .mass_prediction import *
from .mass_score import *
class MaskRCNN(FasterRCNN):
    """
        参数:
            backbone (nn.Module): restnet50+fpn的主干特征网络
            num_classes (int): 识别物体的种类.
            min_size (int): 对图片进行预处理的最小长度
            max_size (int): 对图片预处理的最大长度
            image_mean (Tuple[float, float, float]): 图片预处理normalize的均值
            image_std (Tuple[float, float, float]): 方差

            RPN
            rpn_pre_nms_top_n_train (int): training模式下，nms处理之前每张图片保留的proposal数量
            rpn_pre_nms_top_n_test (int): testing模式下，nms处理之前每张图片保留的proposal数量
            rpn_post_nms_top_n_train (int): training模式下，nms处理之后每张图片保留的proposal数量
            rpn_post_nms_top_n_test (int): testing模式下，nms处理之后每张图片保留的proposal数量
            rpn_nms_thresh (float): rpn中nms处理的阈值
            rpn_fg_iou_thresh (float): rpn中计算损失时选择正样本的iou阈值  大于即为正样本
            rpn_bg_iou_thresh (float): rpn中计算损失时选择负样本的iou阈值  小于即为负样本
            rpn_batch_size_per_image (int): rpn计算损失的选择的样本总数
            rpn_positive_fraction (float): 样本总数中，正样本所占的比率
            rpn_score_thresh (float): rpn中在nms处理之前进行小概率滤除的阈值

            faster
            box_score_thresh (float):testing 模式下，进行nms处理之前的小概率滤除
            box_nms_thresh (float): testing模式下，进行nms处理的阈值
            box_detections_per_img (int): 每张图片允许的最大预测框的个数
            box_fg_iou_thresh (float): faster分支计算误差时，划分正样本的阈值
            box_bg_iou_thresh (float): 划分负样本的阈值
            box_batch_size_per_image (int): fasterrcnn计算误差时采样的样本总数
            box_positive_fraction (float): 正样本的比例
            representation_size (int) : faster head全连接最后的输出通道数
            bbox_reg_weights (Tuple[float, float, float, float]): 计算边界框回归参数是xywh的权重

        """

    def __init__(
            self,
            backbone,
            num_classes=None,
            parameters = None,
            # transform参数
            min_size=800,
            max_size=1333,
            image_mean=[0.598, 0.621, 0.67],  #[0.402, 0.379, 0.330]     [0.485, 0.456, 0.406],
            image_std=[0.183, 0.158, 0.166],   #[0.229, 0.224, 0.225],

            # RPN参数
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,
            # faster 参数
            
            box_score_thresh=0.6,   #只在testing模式下启用以及训练过程中的验证模式下启用
            box_nms_thresh=0.5,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
            representation_size = 1024,    # faster分支部分两个全连接层的输出通道数
            bbox_reg_weights=None
            
            ):
             
        super().__init__(backbone, num_classes, # 主干网络和预测类别个数

        # RPN 部分的相关参数
        # rpn中在nms处理前保留的proposal数（training,testing）
        rpn_pre_nms_top_n_train, 
        rpn_pre_nms_top_n_test, 
        # rpn中在nms处理后保留的proposal数   
        rpn_post_nms_top_n_train, 
        rpn_post_nms_top_n_test,  
        rpn_nms_thresh,  # rpn中进行nms处理时使用的iou阈值
        rpn_fg_iou_thresh, # rpn计算损失时，采集正负样本设置的阈值
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image, # rpn计算损失时采样的样本数
        rpn_positive_fraction,   # 以及正样本占总样本的比例  
        rpn_score_thresh, # rpn中对取出的proposal进行小概率滤除的阈值，之后进行nms处理  

        # faster分支参数      
        box_score_thresh, # nms处理之前移除低目标概率  
        box_nms_thresh,  #fast rcnn中进行nms处理的阈值
        box_detections_per_img, # 对预测结果根据score排序取前100个目标

        # fast rcnn计算误差时，采集正负样本设置的阈值
        box_fg_iou_thresh, # 上限（大于即为正样本）
        box_bg_iou_thresh,  # 下限（小于即为负样本）
        box_batch_size_per_image, # fast rcnn计算误差时采样的样本数
        box_positive_fraction,  # 以及正样本占所有样本的比例
        representation_size,    # faster分支部分两个全连接层的输出通道数
        bbox_reg_weights  # 边界框回归时 xywh所占的比重
    
        )

        self.backbone = backbone
        # 对数据进行标准化，缩放，打包成batch等处理部分
        self.transform = GeneralizedRCNNTransform(min_size, max_size,image_mean,image_std)
        # backbone网络的特征层输出通道数 256
        out_channels = backbone.out_channels

        # mask分支 roialign池化层
        mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        
        # mask分支 四个卷积层
        mask_layers = (256, 256, 256, 256) # 四个卷积层的通道数
        mask_dilation = 1
        mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)
        
        # mask分支  最后的预测头
        mask_predictor_in_channels = 256  # 预测头的输入通道数
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

   
        # mask 分支
        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor

        #############################################
        # mass predictionf 分支
        mass_layers = (256, 128, 128, 64) # 四个卷积层的通道数
        fc_output = 1024     # 第一个全连接层的输出大小
        # output_mass_num = 4  # 最后一层的输出大小
        #mass_ind = [2,3]         # 物料所对应的目标编号
        ################################################################
        # scorehead = None
        scorehead = MassScoreHead(out_channels,mass_layers)
        self.mass_score_pre = MassScorePredcition(mass_layers[3],fc_output,parameters.mass_number,cat_style = parameters.cat_style)


        mass_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        masshead = MassHead(out_channels,mass_layers)  # mass 分支四个卷积层
        massprediction = MassPredcition(mass_layers[3]*7*7,fc_output,parameters.mass_number)
        self.mass = MassBranch(scorehead,mass_roi_pool,masshead,massprediction,parameters)
      
 

        
    def forward(self,image,target= None,epoch = 0):

        original_image_sizes = []
        for img in image:
            val = img.shape[-2:]
            assert len(val) == 2  # 防止输入的是个一维向量
            original_image_sizes.append((val[0], val[1]))

        if self.training == True:
            if target== None:
                raise ValueError("训练模式下，目标信息不能为空")
      
        image_new,target_new = self.transform(image,target)
        
        features = self.backbone(image_new.tensors)

        # 将特征层以及标注target信息传入rpn中
        # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
        # 每个proposals是绝对坐标，且为(x1, y1, x2, y2)格式
        proposals, proposal_losses = self.rpn(image_new, features, target_new)
        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分
        detections, detector_losses,mask_pos_proposals,pos_labels,pos_mask = self.roi_heads(
            features, proposals, image_new.image_sizes, target_new)
        
         
        #################### mass 分支
        detections,mass_loss,score,score_input = self.mass(features,mask_pos_proposals,pos_labels,detections,pos_mask,image_new.image_sizes,target_new)

        ###################### mass score分支
        if self.training: 
            if score!= None and score_input!= None:
                detections,score_loss = self.mass_score_pre(score_input,detections,score)
            else:
                score_loss = {}
        else:
            if  score_input!= None:
                detections,score_loss = self.mass_score_pre(score_input,detections,score)
            else:
                score_loss = {}
        
        #  对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        detections = self.transform.postprocess(
            detections, image_new.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(mass_loss)
        losses.update(score_loss)

        if self.training :
            return losses
        else:
            return detections



# mask分支 四个卷积层
class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        参数:
            in_channels (int): 输入的通道数 256
            layers (tuple): fcn每层的通道数
            dilation (int): 卷积填充padding = 1
        """
        d = OrderedDict()
        next_feature = in_channels 

        for layer_idx, layers_features in enumerate(layers, 1):
            d[f"mask_fcn{layer_idx}"] = nn.Conv2d(next_feature,
                                                  layers_features,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=dilation,
                                                  dilation=dilation)
            d[f"relu{layer_idx}"] = nn.ReLU(inplace=True)
            next_feature = layers_features

        super().__init__(d)
        
        # 初始化权重
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

# mask分支 预测头
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0))
        ]))
        # 初始化权重   
        # named_parameters返回网络层的名字与迭代器
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
