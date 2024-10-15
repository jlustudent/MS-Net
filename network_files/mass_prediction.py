import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from network_files import det_utils,boxes
from network_files.Attention import *
from typing import List, Dict
import matplotlib.pyplot as plt
# mass分支的损失函数
# 输入参数为图片的真实标注数据gt_mass(未经过归一化之后的数据)
# 网络的预测mass输出
def Mass_Loss(gt_mass,prediction_mass,Mass_idx_len):
    gt_mass_proposal = torch.cat(gt_mass,dim = 0).view(-1,4)   # [all_mass_proposal,4]

    device = prediction_mass.device
    gt_mass_proposal = gt_mass_proposal.to(device) # 修改变量的设备类型
    # 网络输出的允许梯度传递，标签数据禁止梯度传递
    gt_mass_proposal.requires_grad_(False) 
    prediction_mass.requires_grad_(True)

    # 损失函数 
    mass_loss = det_utils.balanced_l1_loss(prediction_mass,gt_mass_proposal)/sum(Mass_idx_len)

    return mass_loss

def calculate_score(prediction,target,mass_prediction,gt_mass_score):  # 真值与mass_head预测值
    # target = torch.cat(target,dim = 0).view(-1,4)
    # gt_mass_score= torch.cat(gt_mass_score,dim = 0).view(-1,4)
    device = prediction.device
    zero = torch.tensor(0.).to(device)
    score = []
    for p,t,mp,mt in zip(prediction,target,mass_prediction,gt_mass_score):
        input  = torch.cat((p.unsqueeze(0),t.unsqueeze(0)),dim = 0)
        input_m  = torch.cat((mp.unsqueeze(0),mt.unsqueeze(0)),dim = 0)

        tesnor_ratio = p.unsqueeze(0)/t.unsqueeze(0)
        ratio_tensor = torch.where(tesnor_ratio < 0, zero, tesnor_ratio)  # 小于0的值设为0
        ratio_tensor = torch.where(ratio_tensor > 2, zero, ratio_tensor)  # 大于2的值设为2
        ratio_tensor = torch.where((ratio_tensor > 1) & (ratio_tensor < 2), 2 - ratio_tensor, ratio_tensor) 

        tesnor_mass = mp.unsqueeze(0)/mt.unsqueeze(0)
        mass_tensor = torch.where(tesnor_mass < 0, zero, tesnor_mass)  # 小于0的值设为0
        mass_tensor = torch.where(mass_tensor > 2, zero, mass_tensor)  # 大于2的值设为2
        mass_tensor = torch.where((mass_tensor > 1) & (mass_tensor < 2), 2 - mass_tensor, mass_tensor) 
        
        ratio_min = torch.min(ratio_tensor)
        mass_min = torch.min(mass_tensor)
        ratio_max = torch.max(ratio_tensor)
        mass_max = torch.max(mass_tensor)

        ratio = (ratio_min+ratio_max)/2
        mass = (mass_min*0.6)+(mass_max*0.4)
        CC = torch.corrcoef(input)
        CC_m = torch.corrcoef(input_m)

        # if ratio==0 or CC[0,1]<=0 :
        #     s = CC_m[0,1]*mass
        # el
        if mass==0 or CC_m[0,1]<=0:
            s = CC[0,1]*ratio
        else:
            s = CC_m[0,1]*mass

        s = torch.clamp(s, min=0, max=1)   
        score.append(s.unsqueeze(0))

    score = torch.cat(score,dim = 0).unsqueeze(1)

    return score.detach()

#  mass分支的四个卷积层
class MassHead(nn.Module):

    def __init__(self, in_channels,layer_channels):
        '''
            in_channels (int): 输入的通道数 256
            layers (tuple): fcn每层的通道数 [256,128,128,64]

        '''
        super(MassHead,self).__init__()
        # 第一个卷积层   通道数不变  [257,14,14]     
        self.Con1 = nn.Conv2d(in_channels+1,layer_channels[0],kernel_size= 3,padding= 1)
        self.Rel1 = nn.ReLU(inplace=True)

        self.Con11 = nn.Conv2d(layer_channels[0],layer_channels[0],kernel_size= 3,padding= 1)
        self.Rel11 = nn.ReLU(inplace=True)
        self.Con12 = nn.Conv2d(layer_channels[0],layer_channels[0],kernel_size= 3,padding= 1)
        self.Rel12 = nn.ReLU(inplace=True)


        # 第二个卷积层   通道数减半 [128,14,14]
        self.Con2 = nn.Conv2d(layer_channels[0],layer_channels[1],kernel_size= 3,padding= 1)
        self.Rel2 = nn.ReLU(inplace=True)

        # 第三个卷积层   通道数不变，尺寸减半 [128,7,7]
        self.Con3 = nn.Conv2d(layer_channels[1],layer_channels[2],kernel_size= 3, stride=2 ,padding= 1)
        self.Rel3 = nn.ReLU(inplace=True)

        # 第四个卷积层   通道数减半 [64,7,7]
        self.Con4 = nn.Conv2d(layer_channels[2],layer_channels[3],kernel_size= 3,padding= 1)
        self.Rel4 = nn.ReLU(inplace=True)

                # 初始化权重
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self,x):
        
        # 第一层  256 14 14
        x = self.Con1(x)
        x = self.Rel1(x)

        x = self.Con11(x)
        x = self.Rel11(x)
        x = self.Con12(x)
        x = self.Rel12(x)

        # 128 14 14
        x = self.Con2(x)
        x = self.Rel2(x)

        # 128 7 7
        x = self.Con3(x)
        x = self.Rel3(x)

        # 64 7 7
        x = self.Con4(x)
        x = self.Rel4(x)

        return x
    
# mass 分支预测头
class MassPredcition(nn.Module):
    def __init__(self,in_channels,linear_size,mass_num):
        '''
            in_channels (int): 输入的通道数 64*7*7
            linear_size:第一个全连接层的输出大小 1024
            mass_num: 为最终预测的输出数量（第二个全连接层的输出大小） 4

        '''
        super(MassPredcition,self).__init__()

        self.fc1 = nn.Linear(in_channels,linear_size)
        self.fc2 = nn.Linear(linear_size,linear_size)
        self.mass_score = nn.Linear(linear_size,mass_num)

    def forward(self,x):
        # 第一步对数据进行展平处理
        x = x.flatten(start_dim=1)  # 数据展平处理[all_num_proposal,64*7*7]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.mass_score(x)

        return x

# mass 分支类
class MassBranch(nn.Module):
    def __init__(self,mass_score,mass_roi,mass_head,mass_prediction,para) -> None:
        super().__init__()
        """
        mass_roi 为 roialign池化层
        mass_head 为mass分支的四个卷积层
        mass_prediction 为最终的预测输出
        """
        self.sensor = para.senor
        self.mass_score = mass_score
        self.mass_roi_pool =mass_roi
        self.masshead = mass_head
        self.massprediction =mass_prediction
        self.mass_label = para.mass_label                  # 物料对应的label标签类别
        self.truck_label = para.truck_label                      # 车辆对应的label标签类别
        self.initial_para = torch.tensor(para.initial_para)  # 矿卡初始质量参数
        self.mask_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        # self.Avgpool = nn.AvgPool2d(2,stride=2)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.cpa_cca = CPA_CCA_block(input = 256)
        
    def find_condiction(self,label,result1,result2 = None):
        condition = torch.zeros_like(label, dtype=torch.bool)
        for value in result1:
            condition |= (label == value)
        idx = torch.where(condition)[0]

        if result2:
            condition2 = torch.zeros_like(label, dtype=torch.bool)
            for value in result2:
                condition2 |= (label == value)
            Idx = torch.where(condition2)[0]
            return idx,Idx
        else:
            return idx
    def test_positive(self,img_num,label,proposal,idx,target,class_label):

        label_idx = self.find_condiction(target[img_num]["labels"],class_label)  
        target_box = target[img_num]["boxes"][label_idx]
        IoU = boxes.box_iou(target_box,proposal[idx])
        max_idx = torch.argmax(IoU,dim=0)
        keep = torch.where(target[img_num]["labels"][label_idx[max_idx]] == label[idx])
        box = proposal[idx[keep]]
        return box, idx[keep],max_idx[keep]
    
    def forward(self,feature,pos_proposals,pos_label,prediction_results,pos_mask,image_size,target):
        """
        feature 为fpn层的输出特征层
        pos_proposals 在training模式下为mask分支中用来计算损失的正样本
        pos_label  在training模式下为pos_prosposal对应的gt类别
        prediction_results 在testing模式下使用,记录了faster分支输出的box label score信息
        image_size 为经过tranfrom处理之后的图片大小（此时每张图片的size不一定相同，之后经过batch打包处理之后的大小才相同）
        target 目标信息
        """
        mass_proposal = []
        truck_proposal = []
        mass_data = []
        score_ratio_data = []
        lable_mass = []
        Mass_idx = []
        Mass_idx_len = []
        detections = prediction_results
        loss_mass = {}
        img_num = 0
        len_mask = 0
        if self.training:
            # training模式下使用512个proposal中的正样本中的预测mass的框
            # for t in target:
            #     assert t["mass"] is not None
            # 将列表转换为张量
            

            
        # 提取正样本proposal中预测mass的proposals
            for label,proposal in zip(pos_label,pos_proposals):

                
                mass_idx,truch_idx = self.find_condiction(label,self.mass_label,self.truck_label)    # 获取proposal中预测mass类的索引号
                if min(mass_idx.shape) == 0 or min(truch_idx.shape) == 0:
                    img_num+=1
                    len_mask+=label.shape[0]
                    
                else:
                    # label_idx = self.find_condiction(target[img_num]["labels"],self.truck_label)  # 矿卡的边界框索引
                    # target_box = target[img_num]["boxes"][label_idx]
                    # truck_IoU = boxes.box_iou(target_box,proposal[truch_idx])   # 计算矿卡预测框与真值框的IoU
                    # max_idx = torch.argmax(truck_IoU,dim=0)
                    # # 保留预测矿卡边界框预测类别与目标编辑框类别一致的
                    # keep = torch.where(target[img_num]["labels"][label_idx[max_idx]] == label[truch_idx]) 
                    # max_idx = max_idx[keep]
                    # # 返回经过检测的truckbox（其预测类别与计算iou重合最大的真实框类别一致）
                    # truck_boxes = proposal[truch_idx[keep]]
                    
                    # 返回经过检测的massbox（其预测类别与计算iou重合最大的真实框类别一致）
                    mass_boxes,mass_idx,_ = self.test_positive(img_num,label,proposal,mass_idx,target,self.mass_label)
                    truck_boxes,_,max_idx = self.test_positive(img_num,label,proposal,truch_idx,target,self.truck_label)
                    
                    mass_truck = boxes.box_iou(truck_boxes,mass_boxes)

                    if truck_boxes.shape[0]>=mass_boxes.shape[0]:
                        _,idx = torch.max(mass_truck,0)
                        truck_boxes = truck_boxes[idx]
                        max_idx = max_idx[idx]
                        
                    else:
                        _,idx = torch.max(mass_truck,1)
                        mass_idx = mass_idx[idx]

                    Mass_idx.append(mass_idx+len_mask)
                    lable_mass.append(label[mass_idx])
                    len_mask+=label.shape[0]
                    mass_data.append(target[img_num]["mass"][max_idx])
                    score_ratio_data.append(target[img_num]["ratio"][max_idx])
                    truck_proposal.append(truck_boxes)

                    Mass_idx_len.append(len(mass_idx))         # 记录每张图片中预测mass的propoal的数量
                    img_num+=1
            if  len(Mass_idx) !=0:    
                lable_mass = torch.cat(lable_mass,dim = 0)
                Mass_idx = torch.cat(Mass_idx,dim = 0)
                
            
        else:
            
            # testing 模式下使用faster分支预测出来的box框 当中对卡车框选的box框
            Mass_idx_len = []
            gt_mass_idx,gt_truck_idx = self.find_condiction(prediction_results[0]["labels"],self.mass_label,self.truck_label)
            if min(gt_truck_idx.shape) == 0 or min(gt_mass_idx.shape) == 0:
                loss_mass = {}
                score = None
                score_input = None
                return detections ,loss_mass,score,score_input
            else:
                
                Mass_idx = gt_mass_idx
                mass_proposal = [p["boxes"][gt_mass_idx] for p in prediction_results]
                truck_proposal = [p["boxes"][gt_truck_idx] for p in prediction_results]

                mass_truck = boxes.box_iou(truck_proposal[0],mass_proposal[0])

                if truck_proposal[0].shape[0]>=mass_proposal[0].shape[0]:
                    _,idx = torch.max(mass_truck,0)
                    truck_proposal[0] = truck_proposal[0][idx]
                        
                else:
                    _,idx = torch.max(mass_truck,1)
                    Mass_idx = Mass_idx[idx]

                lable_mass = prediction_results[0]["labels"][Mass_idx]
                Mass_idx_len.append(len(Mass_idx))

        

        if all(tensor.numel() == 0 for tensor in truck_proposal) or sum(Mass_idx_len) ==0:  #如果所有的元素都为零，表明没有存在预测物料的边界框
            loss_mass = {}
            score = None
            score_input = None
            return detections ,loss_mass,score,score_input
        else:

            # mass_proposal = torch.tensor([[],[]])
            # roi align层
            mass_feature = self.mass_roi_pool(feature,truck_proposal,image_size)

            # 注意力机制添加
            # mass_feature = self.cpa_cca(mass_feature)

            # 拼接mask 通道数由256变为257
            mass_mask = pos_mask[Mass_idx,lable_mass].reshape(-1,1,28,28)
            # mass_mask = self.mask_conv(mass_mask)
            # mass_mask = self.Avgpool(mass_mask)
            mass_mask = self.maxpool(mass_mask)
            mass_feature = torch.cat((mass_feature,mass_mask),dim = 1)

            #########################################################################
            score_input = self.mass_score(mass_feature)   # MassScorePredcition类的输入
            # score_input = None   # MassScorePredcition类的输入
            ############################################################################
            
            # 四个卷积层
            mass_feature = self.masshead(mass_feature)
            ############################################################################
            # score_input = mass_feature
            #最后预测层
            mass_feature = self.massprediction(mass_feature)
            
            mass_prediction = None

            if self.training:
                
                # 提取每张图片的真实mass数据
                device  = mass_feature.device
                gt_mass = [(t-self.initial_para.to(t.device))
                               /(self.sensor-self.initial_para.to(t.device)) for T in mass_data for t in T ]
                score_ratio = [s for S in score_ratio_data for s in S]
                gt_mass_ratio = [x * y for x, y in zip(gt_mass, score_ratio)]
                # mass_score = [t for T in mass_data for t in T ]
                gt_mass_score = [m*(self.sensor-self.initial_para.to(device))+self.initial_para.to(device)
                                  for m in gt_mass]
                loss = Mass_Loss(gt_mass_ratio,mass_feature,Mass_idx_len)
                loss_mass = {"loss_mass":loss}
                
                # mass_prediction = mass_feature
                mass_prediction = mass_feature*(self.sensor-self.initial_para.to(device))+self.initial_para.to(device)
                score = calculate_score(mass_feature,gt_mass,mass_prediction,gt_mass_score)
                detections: List[Dict[str, torch.Tensor]] = [{"weights":None,"mass":None}]
                detections[0]["weights"] = mass_feature
                detections[0]["mass"] = mass_prediction  
            else:
                device  = mass_feature.device
                # mass_prediction = mass_feature
                mass_prediction = mass_feature*(self.sensor-self.initial_para.to(device))+self.initial_para.to(device)
                prediction_results[0]["mass"] = mass_prediction
                prediction_results[0]["weights"] = mass_feature
                detections = prediction_results
                score = None

            return detections,loss_mass,score,score_input



# target = torch.cat(target,dim = 0).view(-1,4)   # [all_mass_proposal,4]
#     device = prediction.device
#     target = target.to(device) # 修改变量的设备类型
#     # 网络输出的允许梯度传递，标签数据禁止梯度传递
#     target.requires_grad_(False) 
#     Prediction = prediction.detach().relu()
#     mass_percent = torch.abs(Prediction)/torch.abs(target)
#     mass_error = torch.zeros_like(mass_percent)
#     #  寻找大于1小于2的参数
#     for idx in range(mass_percent.size()[0]):
#         indices_1_2 = (mass_percent[idx]>1)&(mass_percent[idx]<2)
#         indices_up_2 = (mass_percent[idx]>=2)
#         mass_percent[idx][indices_1_2] = 2-mass_percent[idx][indices_1_2]
#         mass_percent[idx][indices_up_2] = 0
#         mass_error[idx] = 1-mass_percent[idx]
#     # error = torch.clamp(torch.abs(prediction-target)/torch.abs(target),min = 0,max = 1)
#     # err_2 = torch.square(error)
#     # RMSE = 1-torch.sqrt(torch.mean(err_2,dim = 1, keepdim=True))
#     max = torch.max(mass_error, dim=1, keepdim=True)[0]
#     min = torch.min(mass_error, dim=1, keepdim=True)[0]
#     average_percent = torch.mean(mass_percent, dim=1, keepdim=True)
#     score = (1-(max-min)/2)*average_percent
#     is_close_to_zero = torch.allclose(mass_percent, torch.zeros_like(mass_percent))





                
