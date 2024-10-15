import torch
import math
from typing import List, Tuple
from torch import Tensor

# 该类的作用是用于从rpn输出的2000个proposal中取出用于计算损失的256个正负样本
# 在roihead中选择512个样本  正样本占0.25
class BalancedPositiveNegativeSampler(object):

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        参数:
            batch_size_per_image (int): 正负样本的总数
            positive_fraction (float): 正样本所占的比例
        """
        self.batch_size_per_image = batch_size_per_image  # rpn 256  roihead 512
        self.positive_fraction = positive_fraction  # rpn 0.5   roihead 0.25

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        参数:
            matched idxs[batch_size,num_anhors]: 
            在rpn记录了anchors为正负样本的代号；1表示正样本，0表示负样本，-1为废弃样本
            在roihead中[batch_size,2000+num_gt]:大于0的为正样本的物体类别编号，0为负样本。（没有废弃样本）

        返回值:
            pos_idx (list[tensor]) 正样本的索引
            neg_idx (list[tensor]) 负样本的索引

        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像的matched_idxs
        for matched_idxs_per_image in matched_idxs:
            # >= 1的为正样本, nonzero返回非零元素索引
            # positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # = 0的为负样本
            # negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            # 指定正样本的数量 128   在roihead中为512的0.25同样为128
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
          
            # 如果正样本数量不够就直接采用所有正样本
            num_pos = min(positive.numel(), num_pos)

            # 指定负样本数量
            num_neg = self.batch_size_per_image - num_pos
            
            # 如果负样本数量不够就直接采用所有负样本
            num_neg = min(negative.numel(), num_neg)


            # 随机选择指定数量的正负样本
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            # 经过打乱之后的正负样本的索引位置
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # 创建全零矩阵[num_anchors] 
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            # 将正样本的位置置为 1
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            # 将负样本的位置置为 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        # rpn返回值[batch_size,num_anchors]
        # roihead返回值[batch_size,2000+num_gt]
        return pos_idx, neg_idx


# 用于计算gt与anchor的回归参数
@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    参数:
        reference_boxes (Tensor): 真是框的信息
        proposals (Tensor): rpn中为batch_size张图片所有anchors的信息
        weights:
    返回值：
        target[batch_size*num_anchors,4] gt相对于anchors的回归参数
    """

    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # unsqueeze(1)表示在1维度上扩展
    # 返回的大小[batch_size*num_anchors,1]
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

 
    # anchors的宽高与中心坐标
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    # anchors对应gt的宽高与中心坐标 
    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    # gt_box相对于anchors的回归参数
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    # 拼接[batch_size*num_anchors,4]
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets

# 该类的作用是记录对rpn中边界框处理的函数
class BoxCoder(object):


    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        参数:
            weights (4-element tuple)：(1.0, 1.0, 1.0, 1.0)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    #在rpn中计算anchor及其对应的gtbox的回归参数
    # 在roihead中计算挑选出的正负样本proposal【b，512】与对应的gtbox的回归参数  在这里面num_anchors = 512
    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        结合anchors和与之对应的gt计算regression参数
        参数:
            reference_boxes: List[Tensor] 每个proposal/anchor对应的gt_boxes
            proposals: List[Tensor]： anchors/proposals

        Returns: regression parameters

        """
        # 统计每张图像的anchors个数，方便后面拼接在一起处理后在分开
        # reference_boxes和proposal数据结构相同，两张图片的数量进行拼接
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0) # [batch_size*num_anchors,4]
        proposals = torch.cat(proposals, dim=0)  # [batch_size*num_anchors,4]

        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)

        # 输入的大小 [batch_size*num_anchors,4]
        # 返回值target [batch_size*num_anchors,4]为gt相对于anchors的回归参数 tx ty tw th
        targets = encode_boxes(reference_boxes, proposals, weights)

        # 之后对其根据每张图片的anchors数量进行分割[batch_size，num_anchors,4]
        return targets.split(boxes_per_image, 0)

    # 将网络的预测输出加到需要处理的box上  返回的[num,1,4] xyxy
    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """

        参数:
            rel_codes: 经过拼接之后的特征层预测的所有边界框回归参数[num,4]   roihead中[b*num_proposal,num_class+1,4]
            boxes: 传入的信息为anchors或者proposals

        Returns:

        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]  #获取每张图片生成的anchors的数量
        concat_boxes = torch.cat(boxes, dim=0)  # 对每张图片的anchor进行拼接 [num,4]

        # 记录所有的anchors数量
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        # 将预测的bbox回归参数应用到对应anchors上得到预测bbox的坐标 [num,4] 坐标为xyxy
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

        # 防止pred_boxes增加维度 [num,1,4]
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes  #[num,1,4]  roihead中[b*num_proposal,num_class+1,4]  xyxy

    # decode函数中的调用函数 将网络的预测调整量加到anchor或者prosposal上
    def decode_single(self, rel_codes, boxes):
        """
        目的是将网络的预测box回归参数加入到anchors上.

        Arguments:
            rel_codes (Tensor): 经过rpn_head处理之后得到的每个anchor的回归参数 [num,4]
            boxes (Tensor): 需要调整的 (anchors/proposals)[num,4]
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor/proposal宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor/proposal高度
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor/proposal中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor/proposal中心y坐标

        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        dx = rel_codes[:, 0::4] / wx   # 预测anchors/proposals的中心坐标x回归参数
        dy = rel_codes[:, 1::4] / wy   # 预测anchors/proposals的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww   # 预测anchors/proposals的宽度回归参数
        dh = rel_codes[:, 3::4] / wh   # 预测anchors/proposals的高度回归参数

        # 限制预测的最大宽高，防止后处理是经过exp()函数产生指数爆炸
        # self.bbox_xform_clip=math.log(1000. / 16)   4.135
        dw = torch.clamp(dw, max=self.bbox_xform_clip)   # 设置dw和dh的一个上下限
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # 将生成的anchors坐标加上特征层的输出，得到proposal的坐标
        # x = tx*w+xa   x为计算出的proposal坐标
        # tx为网络的预测的回归参数，w为调整权重，xa为输入box的坐标(例如输入anchors的坐标)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        #将其转化为左上角和右下角的形式
        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes # [num,4]xyxy


class Matcher(object):
# 在rpn中其call函数返回 每个anchor与所有gt计算iou iou0.7或者最大的gt索引
# 在roihead中 范围为0.5-0.5 即没有废弃样本 大于0.5为正样本，小于则为负样本
    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        参数:
            high_threshold (float): 依据iou大小判断正负样本的上限值
            low_threshold (float): 下限值:
              
            allow_low_quality_matches (bool): if True表示对于某个真实框，如果没有anchor大于high_threshold的话
                    就启用这个真实框对应所有anchor中最大的那个iou.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        self.high_threshold = high_threshold  # rpn 0.7  roihead 0.5
        self.low_threshold = low_threshold    # rpn 0.3  roihead 0.5
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        计算anchors与每个gtboxes匹配的iou最大值，并记录索引，
        iou<low_threshold索引值为-1(负样本)， low_threshold<=iou<high_threshold索引值为-2（废弃样本）
        Args:
            match_quality_matrix (Tensor[float]): [num_gt,num_anchors].是对于一整图片而言

        Returns:
            matches (Tensor[int64]): [num_anchors]记录了每个anchors对应gt的iou最大的gt索引.
        """
        # M x N 的每一列代表一个anchors与所有gt的匹配iou值
        # matched_vals [num_anchors]代表每列的最大值，即每个anchors与所有gt匹配的最大iou值
        # matches[num_anchors]对应最大值所在的索引
        matched_vals, matches = match_quality_matrix.max(dim=0)  
        if self.allow_low_quality_matches:
            all_matches = matches.clone()    # 使用clone后，不共享地址，但是梯度更新共享
        else:
            all_matches = None

        
        # 计算iou小于low_threshold的索引  （负样本的索引）
        below_low_threshold = matched_vals < self.low_threshold

        # 计算iou在low_threshold与high_threshold之间的索引值（废弃样本的索引）
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        # iou小于low_threshold的matches索引置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        # iou在[low_threshold, high_threshold]之间的matches索引置为-2
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches  #[num_anchors]

    # 对某个真实框没有一个iou值大于 high_threshold的情况下启动
    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):

        # 传入的参数 all_matches是matches未修改之前的clone版本
        # match_quality_matrix[num_gt,num_anchors] iou值

        
        # 对于每个gt boxes寻找与其iou最大的anchor，
        # highest_quality_foreach_gt为匹配到的最大iou值
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  

        
        # 寻找每个gt_boxes与其iou最大的anchor索引，一个gt匹配到的最大iou可能有多个anchor
        #  其返回的内容中有两个元素，分别为满足要求的元素所在的行，与列
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )

        # pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1] 值需要列元素
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 保留该anchor匹配gt最大iou的索引，即使iou低于设定的阈值
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]

#  smooth l1损失计算
def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    input 为正样本的网络预测回归参数
    target为对应的真是框回归参数
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

# 平衡L1损失
def balanced_l1_loss(input, target, beta:float=1./(torch.exp(torch.tensor(3))+1), alpha=0.5,gamma:float = 1.5):
    """
    
    Args:
        input (Tensor): 预测框.
        target (Tensor): 真实框坐标.
        beta (float): .
        alpha (float): 0.5.
        
    Returns:
        Tensor: Balanced L1 loss.
    """
    diff = torch.abs(input - target)
    b = torch.exp(torch.tensor(6))-1

    C = (2-0.5*torch.exp(torch.tensor(3)))/b

    balanced_l1_loss = torch.where(diff < beta, alpha/b*(b*diff+1)*torch.log(b*diff+1)-alpha*diff, gamma*diff +C)
    return balanced_l1_loss.sum()

# 平衡L1损失
def balanced_l1_loss_score(input, target, beta:float=1./(torch.exp(torch.tensor(2))+1), alpha=0.5,gamma:float = 1):
    """
    
    Args:
        input (Tensor): 预测框.
        target (Tensor): 真实框坐标.
        beta (float): .
        alpha (float): 0.5.
        
    Returns:
        Tensor: Balanced L1 loss.
    """
    diff = torch.abs(input - target)
    b = torch.exp(torch.tensor(4))-1

    C = (1.5-0.5*torch.exp(torch.tensor(2)))/b

    balanced_l1_loss = torch.where(diff < beta, alpha/b*(b*diff+1)*torch.log(b*diff+1)-alpha*diff, gamma*diff +C)
    return balanced_l1_loss.mean()

