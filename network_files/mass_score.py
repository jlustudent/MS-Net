import torch.nn as nn
import torch
import torch.nn.functional as F
from network_files import det_utils

#  mass_score 分支的四个卷积层
class MassScoreHead(nn.Module):

    def __init__(self, in_channels,layer_channels):
        '''
            in_channels (int): 输入的通道数 256
            layers (tuple): fcn每层的通道数 [256,128,128,64]

        '''
        super(MassScoreHead,self).__init__()
        # 第一个卷积层   通道数减1  [257,14,14]->[256,14,14]       
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
class MassScorePredcition(nn.Module):
    def __init__(self,in_channels,linear_size,mass_num,cat_style = "Conv"):
        '''
            in_channels (int): 输入的通道数 64
            linear_size:第一个全连接层的输出大小 1024
            score_num: 为最终预测的质量分数（第二个全连接层的输出大小）
            cat_style 为质量系数的拼接类型  Conv为卷积拼接   Line为线性拼接 

        '''
        super(MassScorePredcition,self).__init__()

        self.style = cat_style
        if cat_style == "Conv":
            self.fc1 = nn.Linear((in_channels+1)*7*7,linear_size)
            self.fc2 = nn.Linear(linear_size,linear_size)
            self.score = nn.Linear(linear_size,20)
            self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
            # 定义第二次反卷积层（5x5到7x7）
            self.deconv2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

        elif cat_style == "Line":

            self.fc1 = nn.Linear(in_channels*7*7,linear_size)
            self.fc2 = nn.Linear(linear_size,linear_size)
            self.score = nn.Linear(linear_size+mass_num,20)
            
            # self.score = nn.Linear(linear_size,20)
        elif cat_style == "No":
            self.fc1 = nn.Linear(in_channels*7*7,linear_size)
            self.fc2 = nn.Linear(linear_size,linear_size)
            
            self.NM_score = nn.Linear(linear_size,20)
        else:
            raise ValueError("MassScorePrediction类中输入的拼接类型不存在，只有Conv和Line两种")

    def forward(self,x,detections,gt_score = None):
        # 第一步对数据进行展平处理
        mass_predict = detections[0]["weights"]
        loss_mass_score = {}
        if self.style == "Conv":
            cat_conv = self.mass_pre_to_conv(mass_predict)
            cat_conv = self.deconv1(cat_conv)
            cat_conv = self.deconv2(cat_conv)
            x = torch.cat((x,cat_conv),dim = 1)   # 64*7*7->65*7*7
            x = x.flatten(start_dim=1)  # 数据展平处理[all_num_proposal,65*7*7]
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.score(x)
            # x = torch.sigmoid(self.score(x))

            # no_x = x.detach()
            # mask = (x >= 0) & (x <= 1)
            # x[~mask] /= no_x[~mask]


            # 使用20个区域融合相加的方式解封这三个，并修改self.score的输出维度为20
            sequence = torch.arange(0.04, 1, 0.05).expand(x.shape[0],-1).to(x.device) # 4%~99%  共计20
            x = torch.sum(x*sequence,dim = -1).unsqueeze(-1)

            Pre_score = x
        elif self.style == "Line":
            x = x.flatten(start_dim=1)  # 数据展平处理[all_num_proposal,64*7*7]
            x = F.relu(self.fc1(x)) 
            x = F.relu(self.fc2(x))

            x = torch.cat((x,mass_predict),dim = -1)
            x = self.score(x)
        

            # 使用直接预测的方式解封这三行，并修改self.score的输出维度为1
            # no_x = x.detach()
            # mask = (x >= 0) & (x <= 1)
            # x[~mask] /= no_x[~mask]

            # 使用20个区域融合相加的方式解封这三个，并修改self.score的输出维度为20

            sequence = torch.arange(0.04, 1, 0.05).expand(x.shape[0],-1).to(x.device) # 4%~99%  共计20
            x = torch.sum(x*sequence,dim = -1).unsqueeze(-1)
            Pre_score = x

        elif self.style == "No":
            x = x.flatten(start_dim=1)  # 数据展平处理[all_num_proposal,64*7*7]
            x = F.relu(self.fc1(x)) 
            x = F.relu(self.fc2(x))

            x = self.NM_score(x)
        

            # 使用直接预测的方式解封这三行，并修改self.score的输出维度为1
            # no_x = x.detach()
            # mask = (x >= 0) & (x <= 1)
            # x[~mask] /= no_x[~mask]

            # 使用20个区域融合相加的方式解封这三个，并修改self.score的输出维度为20

            sequence = torch.arange(0.04, 1, 0.05).expand(x.shape[0],-1).to(x.device) # 4%~99%  共计20
            x = torch.sum(x*sequence,dim = -1).unsqueeze(-1)
            Pre_score = x
              
        else:
            raise ValueError("MassScorePrediction类中输入的拼接类型不存在，只有Conv和Line两种")
        
        if self.training:

            score_loss = det_utils.balanced_l1_loss_score(Pre_score,gt_score)

            # log_cosh = torch.log(torch.cosh(Pre_score - gt_score))
            # score_loss = torch.sum(log_cosh)
            
            loss_mass_score = {"loss_mass_score":score_loss}
            # loss_mass_score = {}


        else:
            detections[0]["mass_score"] = Pre_score

        return detections,loss_mass_score

    
    def mass_pre_to_conv(self,mass):
        device = mass.device
        Score_conv  = torch.zeros(mass.shape[0],1,3,3).to(device)
        conv  = torch.zeros(3,3)
        for idx in range(mass.shape[0]):
            conv[0,0] = mass[idx][0]
            conv[0,1] = (mass[idx][0]+mass[idx][1])/2
            conv[0,2] = mass[idx][1]
            conv[1,0] = (mass[idx][0]+mass[idx][2])/2
            conv[1,1] = torch.mean(mass[idx])
            conv[1,2] = (mass[idx][1]+mass[idx][3])/2
            conv[2,0] = mass[idx][2]
            conv[2,1] = (mass[idx][2]+mass[idx][3])/2
            conv[2,2] = mass[idx][3]
            Score_conv[idx][0] = conv
        
        return Score_conv
    
if __name__ == "__main__":
    mass = MassScorePredcition(64,1024,4,"Line")
    Score_conv  = torch.randn(3,64,7,7)
    masspre  = torch.randn(3,4)
    x = mass(Score_conv,masspre)
    

