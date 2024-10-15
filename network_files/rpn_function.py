from typing import List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList


#  anchor生成类
class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    anchors生成器
    针对每个特征图上的特征点生成三个大小不同的anchors

    Arguments:
        sizes (Tuple[Tuple[int]]):anchors的大小设定
        aspect_ratios (Tuple[Tuple[float]]): 不同anchors的比例
        二者的长度要一致
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

        # 用来保存产生的anchors模板  [5,3,4]一共五中anchors 每钟有3个形式 
        # 在set_cell_anchors函数中产生
        self.cell_anchors = None       
        self._cache = {}


    #  生成三个比例大小的anchors 
    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        参数:
            scales: anchor的大小
            aspect_ratios: h/w 的比例
            dtype: float32
            device: cpu/gpu
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        # [r1, r2, r3]' * [s1, s2, s3]
        # number of elements is len(ratios)*len(scales)
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

       
        # 生成的anchors模板都是以（0, 0）为中心的, shape [len(ratios)*len(scales), 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()  # round 四舍五入
    
    # 针对5个特征层中每个特征层生成三个比例大小anchors模版
    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        
        # 如果cell_anchors已经被复制了  那就直接退出
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            if cell_anchors[0].device == device:
                return

        # 根据提供的sizes和aspect_ratios生成anchors模板
        # anchors模板都是以(0, 0)为中心的anchor  -》[5,3,4]
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    # 返回值为每个特征层上anchors模板的个数  每层的模板个数一般都为3
    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # 产生5个特征层上的所有网格点的anchors
    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        计算预测特征图对应原始图像上的所有anchors的坐标
        Args:
            grid_sizes: 预测特征矩阵的height和width
            strides: 预测特征矩阵上一步对应原始图像上的步距
        """
        anchors = []
        cell_anchors = self.cell_anchors  # 五个特征层每层有三个anchors模板[5,3,4]
        assert cell_anchors is not None

        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            #将网格点的坐标对应到原图的位置
            # shape: [grid_width] 对应原图上的x坐标(列) 
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的y坐标(行)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width*grid_height, 4]
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        # anchors的返回值为[5,num_anchors,4]num_anchors为每个特征层产生的anchors总数
        return anchors  # List[Tensor(all_num_anchors, 4)]


    #将grid_anchor函数产生的所有特征层的anchor返回（此处的anchor大小坐标是相对于原图的）
    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)
        
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)   #其返回的是5个特征层的网格点对应原图的像素点的anchors坐标
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]

        # 获取每个预测特征层的尺寸(height, width)->五个元素  每个元素为对应特征层的宽高
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # 获取输入图像的height和width，tensor表示输入的图像
        image_size = image_list.tensors.shape[-2:]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # 计算特征层上的一步等于原始图像上的步长（即特征图上一个网格代表原图上的多少个像素）[5,2]
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        # 根据提供的sizes和aspect_ratios生成anchors模板
        self.set_cell_anchors(dtype, device)

        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        # 使用这个函数仅作为帮助 TorchScript 进行静态类型检查和优化的工具，对于普通的Python代码执行并不会有实际的影响
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])

        # 遍历一个batch中的每张图像宽高
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)

        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        # anchors是个list，每个元素为一张图像的所有anchors信息
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        
        self._cache.clear()
        return anchors

# 对backbone 输出的feature进行物体回归和边界框回归
class RPNHead(nn.Module):
    """
    通过将fpn输出的feature分别使用一个1*1的卷积层和一个3*3的卷积层处理
    得到对是否为物体的输出，以及anchors的回归参数
    通过滑动窗口计算预测目标概率与bbox regression参数

    Arguments:
        in_channels: feature的通道数，设定为256个
        num_anchors:  特征层anchor模板的个数 一般为3
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        # 对该网络结构进行初始化
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        # 输入的变量为feature  list形式内有5个元素
        # 返回值：
        #       logits【list】为每个特征层每个网格点上的三个anchor是否为物体
        #       bbox_reg【list】为每个特征层每个网格点上的三个anchor的边界框回归参数
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

# 对rpn——head的输出进行重组
def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    调整tensor顺序，并进行reshape
    参数:
        layer: 单个预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position 3
        C: classes_num or 4(bbox coordinate) 1
        H: height
        W: width

    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C,  H, W)
    # 调换tensor维度
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer

# 对head处理之后的特征层进行拼接，全部拼接在一起[num,1] [num,4]
def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    对box_cla和box_regression两个list中的每个预测特征层的预测信息
    的tensor排列顺序以及shape进行调整 -> [N, -1, C]
    参数:
        box_cls: 每个预测特征层上的预测目标概率
        box_regression: 每个预测特征层上的预测目标bboxes regression参数

    Returns:

    """
    box_cls_flattened = []
    box_regression_flattened = []

    # 遍历每个预测特征层
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width]
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景
        N, AxC, H, W = box_cls_per_level.shape
        # # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        # anchors_num_per_position 每层anchor模板的数量3
        A = Ax4 // 4
        # classes_num   目标或背景  一般为1
        C = AxC // A

        # 对feature中每层的输出进行reshape操作[N, AxC, H, W]->[N, H*W*A, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, Ax4, H, W]->[N, H*W*A, 4]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    # 对特征层所有的物体预测输出做拼接，将所有的拼接在一起[num,1]
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  
    #  box_regression[num,4]
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

## ******************************
## RPN主干网络
class RegionProposalNetwork(torch.nn.Module):
    """
    实施区域提案网络 (RPN).

    参数:
        anchor_generator (AnchorGenerator): anchors生成器
        head (nn.Module): 计算对象性和回归增量的模块
        fg_iou_thresh (float) = 0.7: rpn计算损失正样本的阈值.
        bg_iou_thresh (float) = 0.3: RPN计算损失负样本的阈值.
        batch_size_per_image (int) = 256: RPN计算损失采样的样本数
        positive_fraction (float) = 0.5:  采样样本数中正样本的比例
        pre_nms_top_n (Dict[str]): 记录了RPN中对数据进行nms处理之前保留的proposals数量
                包含两部分 training = 2000个  testing = 1000个
        post_nms_top_n (Dict[str]): 为nms处理之后的proposal数量，也包含两部分，与上述一样
        nms_thresh (float): RPN中进行nms处理时使用到的iou阈值
        score_thresh 表示经过rpn挑选出来的未经过nms处理打的pre_nms_top_n 个proposal,进行小概率滤除的概率阈值
        

    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        # anchors生成器
        self.anchor_generator = anchor_generator

        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # 在训练过程中使用
        # 计算anchors与真实bbox的iou
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh       #预设为0.7
        self.score_thresh = score_thresh
        self.min_size = 1.

    # 针对训练模型测试模式输出不同的proposal保留的个数
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    # 对生成的anchors进行处理，划分正负样本和废弃样本
    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算每个anchors最匹配的gt，并划分为正样本，背景以及废弃的样本
        Args：
            anchors: (List[Tensor]) [batch_size,num_anchors,4]
            targets: (List[Dict[Tensor]) 包含每张图片中的六个元素
        Returns:
            labels: 标记anchors归属类别（1, 0, -1分别对应正样本，背景，废弃的样本）
                    注意，在RPN中只有前景和背景，所有正样本的类别都是1，0代表背景
            matched_gt_boxes：与anchors匹配的gt
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # 提取该图片中的gt_box,原图信息
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算anchors与真实bbox的iou信息 [num_class，num_anchors]都是针对一张图片而言
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)

                # 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                # 取出索引对应的gtbox信息，对于小于零的索引，先按照零处理
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # 用labels_per_image记录正样本的位置为1 
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # 负样本为零
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0

                # 废弃样本为-1
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image) # [batch_size,num_anchors]
            matched_gt_boxes.append(matched_gt_boxes_per_image)  #[batch_size,num_anchors,4]
        return labels, matched_gt_boxes

    # 寻找每个特征层前pre_nms_top_n个最大值的索引
    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        Args:
            objectness: Tensor(每张图像的预测目标概率信息 )
            num_anchors_per_level: List（每个预测特征层上的预测的anchors个数）
        Returns:

        """
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0   # 每一层的偏移量
        # 遍历每个预测特征层上的预测目标概率信息
        # split分割函数
        for ob in objectness.split(num_anchors_per_level, 1):

            num_anchors = ob.shape[1]  # 预测特征层上的预测的anchors个数
            # 将每个特征层内的anchor个数限制在pre_nms_top_n个
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)    

            # 取出ob中前pre_nms_top_n个最大值的索引
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1) # 每层前pre_nms_top_n个最大值的索引
   
    # #####################################################
    # 对调整后的所有proposal,经过nms处理挑选出前pre_nms_top_n个proposal
    # 以及挑选出的proposal对应的网络的物体预测输出概率（经过sigmoid处理后的）
    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        Args:
            proposals: 预测的bbox坐标  [batch_size,num_anchors,4]
            objectness: 预测的目标概率  [num,1]
            image_shapes: batch中每张图片的size信息
            num_anchors_per_level: 每个预测特征层上预测anchors的数目

        Returns:
            box 为proposal中每张图片的前pre_nms_top_n个proposal   [batch_size,pre_nms_top_n,4]
            score 为每个选择出的proposal网络的物体概率预测输出
        """
        num_images = proposals.shape[0]
        device = proposals.device

        # 使用detach后，内存不变，只是梯度更新变为false,即不会计算其梯度也不会进行更新
        objectness = objectness.detach()  # 将梯度信息归零
        objectness = objectness.reshape(num_images, -1)

       
        # levels负责记录anchors属于哪个特征层的索引信息 [num2] num2为一张特征层上的anchors数量
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)

        # 扩展levels [batch_size,num2]
        levels = levels.reshape(1, -1).expand_as(objectness)

        
        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值   
        # 五个特征层最多有5*pre_nms_top_n的anchors索引值   [batch_size,num_proposal]
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):

            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
  
            # 移除小概率boxes
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 经过nms处理后 ，挑选出合格的proposal的索引值
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # 将符合要求的索引限制在2000（post_nms_top_n）个以内，testing模式为1000
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores 

    # rpn中计算网络输出的回归参数，与gt相对于anchors的回归参数的损失值
    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算RPN损失，包括类别损失（前景与背景），bbox regression损失
        参数:
            objectness (Tensor)：预测的前景概率 [num,1]
            pred_bbox_deltas (Tensor)：预测的回归参数 [num,4]
            labels (List[Tensor])：真实的标签 1, 0, -1（batch中每一张图片的labels对应List的一个元素中） [b,num/b]
            regression_targets (List[Tensor])：真实框的回归参数 [b,num/b,4]

        返回值:
            objectness_loss (Tensor) : 类别损失
            box_loss (Tensor)：边界框回归损失
        """
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        # sampled_pos_inds[batch_size,num_anchors] 正样本的位置为1  其余为零 
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # 记录batch_size整图片的正样本索引位置
        # sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]

        # 记录batch_size整图片的负样本索引位置
        # sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # 将所有正负样本索引拼接在一起 [batch_size*256]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        #[num,1]->[num]
        objectness = objectness.flatten()
        # [b,num/b]->[num]
        labels = torch.cat(labels, dim=0)
        # [num,4]
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失（使用正样本计算）
        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失（使用正负样本计算）
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,        # type: ImageList
                features,      # type: Dict[str, Tensor]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
       参数:
            images (ImageList): 经过ImageList类处理之后返回的图像数据
            features (Dict[Tensor]): 主干网络经过fpn输出的5个特征层
            targets (List[Dict[Tensor]): 标签图片上上的真实信息.

        返回值:
            boxes (List[Tensor]): 经过nms处理之后用于后续faster和mask部分使用的proposal
                    其数量依据training和testing不相同.
            losses (Dict[Tensor]): RPN网络的损失. 包含两部分边界损失和物体损失
        """
        
        # features是所有预测特征层组成的OrderedDict
        # 将featrue转化为list[tensor]的形式
        features = list(features.values())

        # 计算每个预测特征层上的预测目标概率和bboxes regression参数
        # objectness和pred_bbox_deltas都是list 
        # 有5个元素（对应每个特征层），
        # objectness每个元素的大小为[batch_size,3,f_h,f_w]
        # pred_bbox_deltas每个元素的大小为[batch_size,3*4,f_h,f_w]

        objectness, pred_bbox_deltas = self.head(features)

        # 生成一个batch图像的所有anchors信息,list(tensor)元素个数等于batch_size
        # 其中每一个tensor【num，4】num都是五个特征层产生的所有anchor数量
        anchors = self.anchor_generator(images, features)

        # 图片的数量  batch_size
        num_images = len(anchors)

        # num_anchors_per_level保存每个预测特征层上的对应的anchors数量
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # 调整内部tensor格式以及shape,将所有特征层对batch_size图片的预测输出全部拼接在一起
        # objectness[num,1]; 记录了每个anchor是前景还是背景
        # pred_bbox_deltas[num,4]  记录了每个anchors的回归参数
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness,
                                                                    pred_bbox_deltas)


        # 将预测的pred_bbox_deltasn[num,4]参数应用到anchors[batch_size,num/batch_size,4]上
        # 得到调整后的proposal 坐标形式为xyxy 左上右下
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)  #[num，1,4]
        # 对proposal进行reshape
        proposals = proposals.view(num_images, -1, 4) #[batch_size，num2,4]

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        # post_nms_top_n在training模式下为2000  test模式下为1000
        # image_sizes记录了batch_size张图片的原始尺寸   
        # [batch_size,post_nms_top_n,4] xyxy 相对于原图大小（image_sizes）的坐标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        # 在训练模式下，需要计算网络输出的边界框回归参数，与真实框与anchors的回归参数之间的损失值
        if self.training:   # 当建立的模型model.train()设定为训练模式下，self.training自动为true
            assert targets is not None
            # 计算每个anchors最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors
            # labels[batch_size,num_anchors]; 记录每张图片的所有anchors是否为正样本，1为正，0为负，-1为废弃
            # matched_gt_boxes[batch_size,num_anchors,4] 每个anchors对应的gtbox，负样本和废弃样本的gtbox使用第一个gt填充
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)

            # regression_targets [batch_size，num_anchors,4]：返回gt相对于anchors的回归参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)

            # 计算物体和边界框损失
            #  objectness[batch_size*num_anchors，1], pred_bbox_deltas[batch_size*num_anchors,4]
            # labels[batch_size，num_anchors，1], regression_targets[batch_size，num_anchors,4]
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
            # boxes [batch_size,post_nms_top_n,4]
            # losses 为Dict类型  记录了物体损失和边界框回归损失
        return boxes, losses
