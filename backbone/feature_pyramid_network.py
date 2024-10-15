from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


class IntermediateLayerGetter(nn.ModuleDict):
    """
    此函数的目的是为了将resnet50拆分成一个个小模型为的就是可以
    获取return_layers中所记录特征层中的输出
    """
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):

       
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out



#   创建 resnet50+FPN的网络结构
class BackboneWithFPN(nn.Module):
    """
    参数:
        backbone (nn.Module) ：主干特征网络 resnet50
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_list 记录resnet50提供给fpn的每个特征层channel
        out_channels (int): fpn每层的输出通道数 256.
        extra_blocks: (LastLevelMaxPool) fpn输出的最后一层进行最大池化处理
    
    """

    def __init__(self,
                 backbone: nn.Module,
                 return_layers=None,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None,
                 ):
        super(BackboneWithFPN,self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()


        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)


        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )

        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class FeaturePyramidNetwork(nn.Module):
    """
    参数:
        见BackboneWithFPN(nn.Module)的初始化函数参数注释
    """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()

        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()

        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()


        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # 对网络中的第一代子模块进行初始化
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks


    # 将resnet网络中layer层的输出的通道数调整到256
    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        
        idx = -1
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    #  使用3*3的卷积，对修改通道数之后的数据进行处理，即fpn对应这一层的输出
    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        通道数不变
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        参数:
            x (OrderedDict[Tensor]): 是resnet50输出的layer1-4，类型为Dict

        返回值:
            results (OrderedDict[Tensor]): layer1-4经过fpn处理后生成的5个feature,通道数都为256.
        """
        # x为字典类型  将x中的key与value分别取出，存入列表中
        names = list(x.keys())
        x = list(x.values())    # list[tensor]

        # result中保存着每个预测特征层
        results = []

        # 将resnet layer4的channel调整到指定的out_channels = 256
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)

            feat_shape = inner_lateral.shape[-2:]  # 获取这个特征层的大小

            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))  # 从头部插入

        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, names)

        # 将fpn特征层计算出的输出，与相应的特征层name记录到Dict字典中
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        
        return out


class LastLevelMaxPool(torch.nn.Module):

    def forward(self, x: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names
