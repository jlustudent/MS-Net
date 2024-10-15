import os
import torch
import torch.nn as nn
from .feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool
import torch.nn.functional as F

'''-------------SE模块-----------------------------'''
#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return y.expand_as(x)
class BasicBlock(nn.Module):
    """搭建BasicBlock模块"""
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,norm_layer = None,Is_SE = False):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 使用BN层是不需要使用bias的，bias最后会抵消掉
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_layer(out_channel)    # BN层, BN层放在conv层和relu层中间使用
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(out_channel)
        self.is_se = Is_SE
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.se = SE_Block(out_channel)   # 选择性创建 SE 模块
    # 前向传播
    def forward(self, X):
        identity = X
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.is_se :  # 如果使用 SE 模块，进行加权
            Y = self.se(Y)

        if self.downsample is not None:    # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(X)

        return self.relu(Y + identity)
 

# 残差结构块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None,Is_SE = False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.is_se = Is_SE
        # self.SE = None
        self.SE = SE_Block(self.expansion*out_channel)
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.is_se:
            SE_out = self.SE(out)
            out = out * SE_out
        out += identity
        out = self.relu(out)

        return out


#   残差网络结构
class ResNet(nn.Module):

    def __init__(self, block, blocks_num, norm_layer=None,is_se = False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

       
        self.in_channel = 64
        self.is_se_block = is_se

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer,Is_SE = self.is_se_block))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet_fpn_backbone(pretrain_path="",
                          trainable_layers=3,
                          Framework = 50,
                          returned_layers=None,
                          extra_blocks=None,
                          Is_se = False):
    """
    搭建resnet50_fpn——backbone
    Args:
        pretrain_path: resnet50的预训练权重，如果不使用就默认为空
        norm_layer: 默认是nn.BatchNorm2d，如果GPU显存很小，batch_size不能设置很大，
                    建议将norm_layer设置成FrozenBatchNorm2d(默认是nn.BatchNorm2d)
                    (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers: 指定训练哪些层结构
        returned_layers: 指定哪些层的输出需要返回
        extra_blocks: 在输出的特征层基础上额外添加的层结构

    Returns:

    """
    if Framework ==50:
        resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3],is_se=Is_se)# 50 3 4 6 3     
    
    elif Framework ==101:
        resnet_backbone = ResNet(Bottleneck, [3, 4, 23, 3],is_se=Is_se)#   101 3 4 23 3
    elif Framework ==34:
        resnet_backbone = ResNet(BasicBlock, [3, 4, 6, 3],is_se=Is_se)# 
    else:
        raise ValueError("必须为 34 50 101")

    #  载入预训练权重文件     
    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} is not exist.".format(pretrain_path)
        # 载入预训练权重
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))

    # 选择需要resnet网络中需要训练的层数
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果要训练所有层结构的话，不要忘了conv1后还有一个bn1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # 将不训练的层数梯度更新设置为false
    for name, parameter in resnet_backbone.named_parameters():
        # 只训练不在layers_to_train列表中的层结构
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    # 对网络的最后一层的输出做最大池化处理，得到P6
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    #  记录resnet中四个layer层哪个层的输出需要返回，不能大于5
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    # in_channel 为layer4的输出特征矩阵channel = 2048
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    # 记录resnet50提供给fpn的每个特征层channel
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # 通过fpn后得到的每个特征层的channel
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

if __name__== "__main__":
    from PIL import Image
    from torchvision import transforms
#     img_path = "306.jpg"
#     original_img = Image.open(img_path).convert('RGB')

# # from pil image to tensor, do not normalize image
#     data_transform = transforms.Compose([transforms.ToTensor()])
#     img = data_transform(original_img)
#     img = torch.unsqueeze(img, dim=0)
    img = torch.randn(1,3,640,640)
    backbone = resnet_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)
    feature = backbone(img)
