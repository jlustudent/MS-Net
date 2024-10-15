from typing import Optional, List, Dict, Tuple
import os
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import roi_align

from . import det_utils
from . import boxes as box_ops

# 计算faster rcnn网络的类别损失和边界框损失
# 传入参数 faster rcnn预测的类别输出，边界框回归参数输出
# labels为512个正负样本的gt标签   regression_target 为512个proposal与gt之间的回归参数
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Faster R-CNN.的损失计算

    参数:
        class_logits : 预测类别概率信息，shape=[b*512, num_classes+1]
        box_regression : 预测边目标界框回归信息[b*512, (num_classes+1)*4]
        labels : 真实类别信息 [b,512]
        regression_targets : 真实目标边界框信息[b,512,4]

    返回值:
        classification_loss (Tensor) 类别损失
        box_loss (Tensor) ：回归参数损失
    """
    # shape转换[b,512]->[b*512]
    labels = torch.cat(labels, dim=0) # 里面为gt的标签以及背景（值为0）
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失信息  （交叉熵损失）
    classification_loss = F.cross_entropy(class_logits, labels)

   
    # 返回标签类别大于0的索引 （正样本的索引）
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    # 返回标签类别大于0位置的类别信息（正样本对应的gt标签）
    labels_pos = labels[sampled_pos_inds_subset]

    
    N, num_classes = class_logits.shape
    # shape [b*512, (num_classes+1),4]
    box_regression = box_regression.reshape(N, -1, 4)

    # 计算边界框损失信息  (使用正样本计算边界框回归损失)
    box_loss = det_utils.smooth_l1_loss(
        # 获取指定索引proposal的指定类别box信息
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    # 计算边界框损失信息  (使用正样本计算边界框回归损失)  balanse L1 loss
    # box_loss = det_utils.balanced_l1_loss(
    #     # 获取指定索引proposal的指定类别box信息
    #     box_regression[sampled_pos_inds_subset, labels_pos],
    #     regression_targets[sampled_pos_inds_subset]
    # ) / labels.numel()

    return classification_loss, box_loss

# 根据网络的输出的label标签将每个box从num_class+1个mask中选出对应类别的mask
def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    根据 CNN 的结果，通过获取与具有最大概率的类对应的掩码
    （大小固定，由 CNN 直接输出）对掩码进行后处理，
    并在 BoxList 的掩码字段中返回掩码。

    参数:
        x (Tensor):[num_pre,num_class+1,28,28]
        labels (list[BoxList]): [num_pre]

    返回值:
        results (list[BoxList]): 
          [b,num_pre,1,28,28] 返回每张图片对应box的标签的mask输出
    """
    # 将预测值通过sigmoid激活全部缩放到0~1之间
    mask_prob = x.sigmoid()

    # num_pre预测出的box个数
    num_masks = x.shape[0]

    # 先记录每张图片中boxes/masks的个数
    boxes_per_image = [label.shape[0] for label in labels]
    # 在将所有图片中的masks信息拼接在一起(拼接后统一处理能够提升并行度)
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    # 提取每个masks中对应预测最终类别的mask
    mask_prob = mask_prob[index, labels][:, None]
    # 最后再按照每张图片中的masks个数分离开
    mask_prob = mask_prob.split(boxes_per_image, dim=0)
    # [b,num_pre,1,28,28] 返回每张图片对应box的标签的mask输出
    return mask_prob

# 从gt_mask上截取proposal位置并进行roi_align池化，用来计算损失
def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
            gt_masks [num_gt，img_w,img_h]; 
            boxes[num_pos,4]; 
            matched_idxs[num_pos]
            M = 28

    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)  # [num_pos,5]5 = id+xyxy
    gt_masks = gt_masks[:, None].to(rois) # [num_gt,1,img_w,img_h]

    # roi_align 类的作用是使用正样本的proposal从gt_mask中截取对应的大小
    # 之后将proposal对应的位置池化到(M, M)的大小，用于计算后续的损失
    #  roi_align 返回的大小[num_pos,28,28]
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]

# 计算 mask分支的损失
def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    参数:
        mask_logits[b*num_pos,num_class+1,28,28] mask分支的最终预测输出
        mask_proposals[b,num_pos，4] 正样本的proposal信息
        gt_masks[b,num_gt，img_w,img_h]
        pos_matched_idxs[b,num_pos] 正样本proposal对应的gt的索引

    返回值:
        mask_loss (Tensor): 
    """

    # 28(FCN分支输出mask的大小)
    discretization_size = mask_logits.shape[-1]

    # 获取每个Proposal(全部为正样本)对应的gt类别 [b,num_pos]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    # 根据Proposal信息在gt_masks上裁剪对应区域做为计算loss时的真正gt_mask
    # 传入的参数分别为：m [num_gt，img_w,img_h]; p[num_pos,4]; i[num_pos]
    # mask_targets[b,num_pos,28,28]真实框mask对应proposal的池化信息
    mask_targets = [project_masks_on_boxes(m, p, i, discretization_size) 
                    for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
                    ]

    # 将一个batch中所有的Proposal对应信息拼接在一起(统一处理提高并行度)[b*num_pos]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0) #[b*num_pos,28,28]

    
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    # 计算预测mask与真实gt_mask之间的BCELoss
    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss

# 处理rpn生成的数据，以及target信息 传入faster Rcnn后续处理部分 其中也包含mask分支
class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign  roialign池化类对象
                 box_head,       # TwoMLPHead类    将池化后的信息展平并经过两个全连接层
                 box_predictor,  # FastRCNNPredictor类  实现对类别和边界框的回归预测

                 # Faster R-CNN 训练过程中使用的
                # faster计算误差时正负样本的阈值都为0.5
                 fg_iou_thresh, bg_iou_thresh,  
                 # 计算误差的采样数 512, 正样本的占比0.25
                 batch_size_per_image, positive_fraction,  
                 bbox_reg_weights,  # 权重，None
                 # Faster R-CNN inference
                 score_thresh,        # 移除低目标概率: 0.05
                 nms_thresh,          # nms处理的阈值: 0.5
                 detection_per_img,   # 预测结果取前: 100个
                 # Mask
                 mask_roi_pool=None, # mask分支的roialign
                 mask_head=None,     # mask分支的四个卷积层
                 mask_predictor=None, # mask分支的最后两个预测输出的卷积
                 ):
        super(RoIHeads, self).__init__()

        # 计算iou的函数
        self.box_similarity = box_ops.box_iou
        
        # 根据iou值返回每个anchor/proposal与gt计算的iou的最大值的gt索引
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)
        
        # 返回正样本和负样本的索引
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25

        # 设置边界框回归参数的权重 xywh 明显xy的权重要高
        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign类
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # testing模式下为0.3
        self.nms_thresh = nms_thresh      # nms阈值: 0.5
        self.detection_per_img = detection_per_img  # 选取个数: 100

        # mask 分支
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

    # 用于检测mask分支是否建立
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
    
    # 为拼接之后的proposal划分正负样本，
    # 返回正负样本对应的gt iou最大值的索引，小于阈值的为负样本暂时定为第0个gt的索引
    # 返回正样本对应的类别号负样本为零
    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        为每个proposal匹配对应的gt_box，并划分到正负样本中
        参数:
            proposals: [b,2000+num_gt,4]
            gt_boxes:[b,num_gt,4] 
            gt_labels: [b,num_gt]

        返回值:
            matched_idxs[b,2000+num_gt]中记录了与gt计算iou最大对应gt索引
            其中大于等于0为正样本  但是其中的负样本也被赋值为0
            labels[b,2000+num_gt]中记正样本对应的类别号，负样本位置为0(表示背景) 
        matched_idxs和labels的正负样本对应
        """
        matched_idxs = []
        labels = []
        # 遍历每张图像的proposals, gt_boxes, gt_labels信息
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                # 计算proposal与每个gt_box的iou重合度 [num_gt,2000+num_gt]
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

                # 计算proposal与每个gt_box匹配的iou最大值 并记录索引[2000+num_gt]
                # 此处不启用allow-low_qulity
                # iou < low_threshold索引值为 -1（负样本）
                #  low_threshold <= iou < high_threshold索引值为 -2（废弃样本）此处不存在
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                # 限制最小值，防止匹配标签时出现越界的情况
                # 注意-1, -2对应的gt索引会调整到0,获取的标签类别为第0个gt的类别（实际上并不是）,后续会进一步处理
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                
                # 获取proposal匹配到的gt对应标签 
                # 获取正样本匹配的标签，负样本的标签为第0个gt的类别
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # 此处对负样本的标签进行修改
                # 将计算iou索引为-1的类别标签设置为0，即背景，负样本
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # 将计算iou索引为-2的类别标签设置为-1，即废弃样本（此处不存在废弃样本）
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        
        #【b,2000+num_gt】
        return matched_idxs, labels
   
    # [b,512] 返回的为每张图片从proposal选出的512个正负样本的索引
    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # labels[b,2000+num_gt]

        # pos[b,2000+num_gt] 正样本的位置为1
        # neg [b,2000+num_gt]负样本的位置为1
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):

            # 记录所有采集样本索引（包括正样本和负样本）
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)

        return sampled_inds  # [b,512] 返回的为选出的512个正负样本的索引


    #对rpn中nms处理之后的proposal[b,2000,4]划分正负样本512个，
    # 返回选出的proposal[b,512,4]、对应gt的索引、对应gt的标签类别、边界框回归信息
    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        """
        划分正负样本，统计对应gt的标签以及边界框回归信息
        list元素个数为batch_size
        参数:
            proposals: rpn预测的boxes [b,2000,4]
            targets:

        返回值：
            proposals[b,512,4]:对rpn输出的proposal提取正负样本，依据与gt的iou值
            matched_idxs[b,512]：每个proposal对应的gt的索引，负样本暂时划分到第0个gt的索引0
            labels[b,512] ： 记录了每个proposal对应的gt的物体类别，负样本为0,表示背景
            regression_targets[b,512,4]：记录每个proposal与对应gt之间的回归参数，负样本都与第0个gtbox计算回归参数

        """
        # 获取数据类型和硬件
        dtype = proposals[0].dtype
        device = proposals[0].device

        # 获取真实标注的boxes以及labels信息 [b,num_gt,4] [b,num_gt]
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]  # gtbox对应的物体种类

        
        # 将gt_boxes拼接到proposal后面 [b,2000+num_gt,4]
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
 
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        # 输入变量的大小[b,2000+num_gt,4][b,num_gt,4] [b,num_gt]
        # matched_idxs[b,2000+num_gt]中记录了与gt计算iou最大对应gt索引
        #其中大于等于0为正样本  但是其中的负样本也被赋值为0
        # labels[b,2000+num_gt]中记正样本对应的类别号，负样本位置为0(表示背景) 
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
       
        # [b,512]返回每张图片在proposal选出的512个正负样本的索引
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []

        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):

            # 获取每张图像的正负样本索引 
            img_sampled_inds = sampled_inds[img_id]

            # 获取对应正负样本的proposals信息
            proposals[img_id] = proposals[img_id][img_sampled_inds]

            # 获取对应正负样本的真实类别信息
            labels[img_id] = labels[img_id][img_sampled_inds]

            # 获取对应正负样本的gt索引信息
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            # 获取每张图片的gtbox信息[num_gt,4]
            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)

            # 保存对应正负样本的gt box信息
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # 根据gt和proposal计算边框回归参数（针对gt的） 
        # 返回值[b,512,4] tx ty tw th
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    #  testing过程中使用
    # [b*num_proposal，num_class+1]
    # [b*num_proposal，(num_class+1)*4]
    # [b,num_prooposal,4]
    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        对网络的预测数据进行后处理，包括
        （1）根据proposal以及预测的回归参数计算出最终bbox坐标
        （2）对预测类别结果进行softmax处理
        （3）裁剪预测的boxes信息，将越界的坐标调整到图片边界上
        （4）移除所有背景信息
        （5）移除低概率目标
        （6）移除小尺寸目标
        （7）执行nms处理，并按scores进行排序
        （8）根据scores排序返回前topk个目标
        参数:
            class_logits: 网络预测类别概率信息
            box_regression: 网络预测的边界框回归参数
            proposals: rpn输出的proposal
            image_shapes: 打包成batch前每张图像的宽高

        Returns:
                all_boxes [num_pre,4]  对图片最终的预测框信息xyxy
                all_scores [num_pre]  每个预测框的预测概率
                all_labels [num_pre]  每个预测框预测的物体种类

        """
        device = class_logits.device
        # 预测目标类别数
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测proposal数量 [b]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        # 根据proposal以及预测的回归参数计算出最终bbox坐标 roihead中[b*num_proposal,num_class+1,4] xyxy
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理 (在num_class+1 维度上进行归一化处理) 
        pred_scores = F.softmax(class_logits, -1)

        
        # 根据每张图像的预测bbox数量分割结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)  # [b,num_proposal,num_class+1,4]
        pred_scores_list = pred_scores.split(boxes_per_image, 0)  # [b,num_proposal,91]

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):

            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # 创建 labels[num_class+1]  内容为[0-2]
            labels = torch.arange(num_classes, device=device)

            # labels[num_proposal，num_class+1]
            labels = labels.view(1, -1).expand_as(scores)

           
            # 移除索引为0的所有信息（0代表背景）
            # shape[num_proposal，num_class+1]->[num_proposal，num_class] 除去背景信息
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]    
            labels = labels[:, 1:]   # [1-90]

             
            boxes = boxes.reshape(-1, 4) # [num_proposal，num_class,4]->[num_proposal*num_class,4]
            scores = scores.reshape(-1)  # [num_proposal*num_class]
            labels = labels.reshape(-1)  # [num_proposal*num_class]

            
            # 移除低概率目标，self.scores_thresh=0.05   testing模式为0.5


            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # 获取scores排在前topk（100）个预测目标
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        
        return all_boxes, all_scores, all_labels

    def forward(self,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None    # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        参数:
            features (List[Dict Tensor]) backbone网络的预测输出5个特征层[5,b,256,w,h]
            proposals (List[Tensor[N, 4]]) 经过rpn中nms处理后挑选出的2000个proposal  [b,2000,4] training模式下
            image_shapes (List[Tuple[H, W]]) 记录了b张图片的原始大小（经过transform处理）
            targets (List[Dict])  
        返回值：
            #  testing模式下：result为字典类型里面保存着 预测框信息，预测框概率，预测物体的种类  training为{}
        """

    

        if self.training:
            # 检查targets的数据类型是否正确
            if targets is not None:
                for t in targets:
                    floating_point_types = (torch.float, torch.double, torch.half)
                    assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                    assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
            else:
                raise ValueError("target should not be None.")
            #对rpn中nms处理之后的proposal[b,2000,4]划分正负样本512个，
            # 返回选出的proposal[b,512,4]、对应gt的索引、对应gt的标签类别、边界框回归信息
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # 将采集样本通过Multi-scale RoIAlign pooling层
        # 传入的参数为backbone输出的feature[5,b,256,w,h]
        # 挑选完正负样本的proposal【b,512,4】 testing[b,max(1000),4]
        # 以及经过transform处理之后的图片大小
        # 返回值为 proposal从feature上截取的大小进行池化  返回的大小为[b*512,256,7,7]
        # testing的返回值为[b*max(1000),256,7,7]
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # 通过roi_pooling后的两层全连接层
        # box_features: [b*512,256,7,7]->[b*512, 1024]
        box_features = self.box_head(box_features)

        # 接着分别预测目标类别和边界框回归参数
        # [b*512, num_class+1][b*512, (num_class+1)*4]
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            # 计算faster rcnn网络的类别损失和边界框损失
            # 传入参数 faster rcnn预测的类别输出，边界框回归参数输出
            # labels为512个正负样本的gt标签   regression_target 为512个proposal与gt之间的回归参数
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:   # testing模式
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        # 检测mask分支是否建立
        if self.has_mask():
            
            mask_proposals = [p["boxes"] for p in result]  # 将最终预测的Boxes信息取出

            if self.training:  # 训练模式下mask_proposals为[]
                # matched_idxs为每个proposal在正负样本匹配过程中得到的gt索引(背景的gt索引也默认设置成了0)   
            
                # batch_size
                num_images = len(proposals)

                mask_proposals = []
                pos_matched_idxs = []
                # 遍历每一张图片
                for img_id in range(num_images):
                    #寻找每张图片的512个proposal中的正样本索引
                    pos = torch.where(labels[img_id] > 0)[0]  # 寻找对应gt类别大于0，即正样本
                    labels[img_id] = labels[img_id][pos]
                    # 记录正样本的proposal以及对应的gt索引
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None
            
            # mask分支的roialign与faster分支的区别在于池化之后的大小变为了14*14
            # mask_proposals[b,num_pos,4]
            # mask_features [b*num_pos,256,14,14]
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)

            # 对mask_features使用四个256通道的卷积层，维度保持不变[b*num_pos,256,14,14]
            mask_features = self.mask_head(mask_features)
            # mask_predictor中对mask_features进行上采样
            # [b*num_pos,256,14,14]->[b*num_pos,256,28,28]
            # 之后使用一个1*1卷积核修改通道数mask_logits[b*num_pos,num_class+1,28,28]
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits 在训练时不能为空")

                # 获取真实标注的mask信息 gt_mask的宽高与image_shapes中记录的大小一致
                gt_masks = [t["masks"] for t in targets]
                # 获取真实标注的物体类别信息
                gt_labels = [t["labels"] for t in targets]

                # 计算mask分支的预测损失
                # mask_logits[b*num_pos,num_class+1,28,28] mask分支的最终预测输出
                # mask_proposals[b,num_pos，4] 正样本的proposal信息
                # gt_masks[b,num_gt，img_w,img_h]
                # pos_matched_idxs[b,num_pos] 正样本proposal对应的gt的索引
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else: # testing模式
                # result中记录faster分支在testing模式下的输出
                #记录网络预测输出的box的类别标签 [num-pre]
                # num_pre为faster网络预测的剥削个数
                labels = [r["labels"] for r in result]

                # mask_logits[num_pre,num_class+1,28,28]
                # mask_probs[b,num_pre,1,28,28]
                mask_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(mask_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)
        #  testing模式下：result为字典类型里面保存着 预测框信息，预测框概率，预测物体的种类
            #  mask_proposals在training模式下保存的是 mask分支中使用的正样本proposals
            # labels在training模式下保存的是 mask分支中使用的正样本proposals所对应的gt类别
            # pos_matched_idxs记录了正样本匹配的gt的索引
        return result, losses,mask_proposals,labels,mask_logits
