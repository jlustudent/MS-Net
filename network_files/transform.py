import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision

from .image_list import ImageList

# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily for paste_mask_in_image
def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor

    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


@torch.jit.unused
def expand_masks_tracing_scale(M, padding):
    # type: (int, int) -> float
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)


def expand_masks(mask, padding):
    # type: (Tensor, int) -> Tuple[Tensor, float]
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # could not import is_tracing(), not sure why
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w):
    # type: (Tensor, Tensor, int, int) -> Tensor

    # refer to: https://github.com/pytorch/vision/issues/5845
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batch, C, H, W]
    # 因为后续的bilinear操作只支持4-D的Tensor
    
    mask = mask.expand((1, 1, -1, -1))  # -1 means not changing the size of that dimension

    # # 生成与原始张量形状相同的噪声张量
    # noise = torch.randn_like(mask)*0.3

    # # 找到大于0的位置
    # mask_idx = mask > 0

    # # 只在大于0的元素位置上添加噪声
    # mask[mask_idx] = mask[mask_idx] + noise[mask_idx]
    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]  # [batch, C, H, W] -> [H, W]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    # 填入原图的目标区域(防止越界)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    # 将resize后的mask填入对应目标区域
    im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])]
    return im_mask


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor

    # pytorch官方说对mask进行expand能够略微提升mAP
    # refer to: https://github.com/pytorch/vision/issues/5845
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape


    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]  # [num_obj, 1, H, W]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self,
                 min_size: int,
                 max_size: int,
                 image_mean: List[float],
                 image_std: List[float],
                 size_divisible: int = 32,
                 fixed_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size      # 指定图像的最小边长范围
        self.max_size = max_size      # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std    # 指定图像在标准化处理中的方差
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size
    
    # 对图像减均值除以方差达到标准化处理
    def normalize(self, image):
        """标准化处理"""
        #获取图片的设备信息
        dtype, device = image.dtype, image.device
        # 均值和方差的类别转换
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None]: shape [3] -> [3, 1, 1]
        #三个元素是因为RGB图像有三个通道
        return (image - mean[:, None, None]) / std[:, None, None]

    #  对图像和标注框做一个缩放处理
    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        将图片缩放到指定的大小范围内，并对应缩放bboxes信息
        参数:
            image: 输入的图片
            target: 输入图片的相关信息（包括bboxes信息）

        Returns:
            image: 缩放后的图片
            target: 缩放bboxes后的图片相关信息
        """
        # image shape  = [channel, height, width]
        h, w = image.shape[-2:]

        im_shape = torch.tensor(image.shape[-2:])
        size: Optional[List[int]] = None
        if self.fixed_size is not None:
            size = [self.fixed_size[1], self.fixed_size[0]]
        else:
            min_size = torch.min(im_shape).to(dtype=torch.float32)  # 获取高宽中的最小值
            max_size = torch.max(im_shape).to(dtype=torch.float32)  # 获取高宽中的最大值
            scale_factor = torch.min(float(self.min_size[-1]) / min_size, self.max_size / max_size)  # 计算缩放比例

        # interpolate利用插值的方法缩放图片
        # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
        # bilinear只支持4D Tensor, 后面使用[0]可以将4D转换为3D
        image = F.interpolate(
            image[None],
            size=size,
            scale_factor=scale_factor.item(),
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False)[0]

        if target is None:
            return image, target

        if "masks" in target:
            mask = target["masks"]
            mask = F.interpolate(
                mask[:, None].float(), size=size, scale_factor=scale_factor,recompute_scale_factor=True
            )[:, 0].byte()  # self.byte() is equivalent to self.to(torch.uint8).
            target["masks"] = mask

        bbox = target["boxes"]
        # 根据图像的缩放比例来缩放bbox
        # image.shape[-2:]是经过interpolate之后的尺寸
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        target["boxes"] = bbox

        return image, target
    
    # 将图像打包成一个batch输入到网络当中
    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
        将一批图像打包成一个batch返回（注意batch中每个tensor的shape是相同的）
        参数:
            images: 输入的一批图片
            size_divisible: 将图像高和宽调整到该数的整数倍

        返回值:
            batched_imgs: 打包成一个batch后的tensor数据
        """
        # ONNX模型是一个开放的神经网络模式，根据这个模式可以在各个模式下进行转化
        # 此处不用管，默认不满足：作用是将模型转化为ONNX模型
        # if torchvision._is_tracing():
        #     return self._onnx_batch_images(images, size_divisible)

        # 分别计算一个batch中所有图片中的最大channel, height, width
        the_list = [list(img.shape) for img in images]
        max_size = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                max_size[index] = max(max_size[index], item)

        stride = float(size_divisible)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        
        # batch_shape[batch, channel, height, width]
        batch_shape = [len(images)] + max_size

        # 创建shape为batch_shape且值全部为0的tensor
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs


    def postprocess(self,
                    result,                # type: List[Dict[str, Tensor]]
                    image_shapes,          # type: List[Tuple[int, int]]
                    original_image_sizes   # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        if self.training:
            return result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                # 将mask映射回原图尺度
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks

        return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]

        images = [img for img in images]
        for i in range(len(images)):
            #获取每一张图片信息
            image = images[i]
            # 图片的标记信息
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
             # 对图像进行标准化处理，减均值除以方差
            image = self.normalize(image) 

            # 对图像和对应的bboxes缩放到指定范围 将图片大小缩放到min_size max_size之间  并对box进行对应的缩放
            image, target_index = self.resize(image, target_index)  
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后的图像尺寸
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, self.size_divisible)  # 将images打包成一个batch
        # 记录resize之后的图片的大小
        image_sizes_list = []

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

# 对边界框缩放
def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    将boxes参数根据图像的缩放情况进行相应缩放

    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    # 计算宽高缩放因子
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]

    ratios_height, ratios_width = ratios
    # gt_box[num-gt, 4]
    # unbind（1）表示在索引为1的维度展开也就是在4的维度展开
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    # 乘以缩放因子后，拼接返回
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


