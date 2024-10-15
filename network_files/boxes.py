import torch
from typing import Tuple
from torch import Tensor





def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    以批处理方式执行非极大值抑制。每个索引值对应一个类别，NMS
    不会在不同类别的元素之间应用。

    参数
    ----------
    boxes : Tensor[N, 4]： 每张图片五个特征层移除小概率之后的box
    scores : Tensor[N]: 每张图片网络输出的物体预测概率（进过sigmoid处理之后的）
    idxs : Tensor[N]： 每个box对应特征层的索引值
    iou_threshold : nms处理的阈值

    Returns
    -------
    keep : Tensor
        经过nms处理之后合格的box的索引
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()

    # 为每一个类别/每一层生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别/层之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = torch.ops.torchvision.nms(boxes_for_nms, scores, iou_threshold)
    return keep

#  将rpn网络中挑选出的proposal的大小限制在图内
def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """
    裁剪预测的boxes信息，将越界的坐标调整到图片边界上

    参数:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) 
        size (Tuple[height, width]): 该大小是经过transform处理之后的图片大小

    Returns:
        clipped_boxes (Tensor[N, 4]) 修改完限制的proposal
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)   # 限制x坐标范围在[0,width]之间
    boxes_y = boxes_y.clamp(min=0, max=height)  # 限制y坐标范围在[0,height]之间

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)

# 计算两个框之间的iou值
def box_iou(boxes1, boxes2):
    """
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): boxes1中每个框与M个boxes2之间的iou值
    """
    # 计算每个框的面积大小
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    #  找到左上角的最大值和右下角的最小值
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    # 计算两个框之间重合部分的面积
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou   # 【N,M】
 
