import torch.nn as nn
import torch
import torch.nn.functional as Fun


#   双感知注意力机制

#  位置注意力机制
class PositionAttention(nn.Module):
    def __init__(self, input,output = 1):
        super(PositionAttention, self).__init__()

        # 四个自适应平均池化层
        self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool_3 = nn.AdaptiveAvgPool2d(3)
        self.avg_pool_6 = nn.AdaptiveAvgPool2d(6)

        #self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.F_to_C   = nn.Conv2d(input, output, 1, bias=False)
        self.F_to_D   = nn.Conv2d(input, input, 1, bias=False)
        self.Input_to_B = nn.Conv2d(input ,output, 1, bias=False)

        self.alpha = nn.Conv2d(input, input, 1, bias=False)

    def forward(self, x):

        out_1 = self.avg_pool_1(x) #[*,*,1,1]
        out_2 = self.avg_pool_2(x) #[*,*,2,2]
        out_3 = self.avg_pool_3(x) #[*,*,3,3]
        out_6 = self.avg_pool_6(x) #[*,*,6,6]

        out_1 = out_1.reshape(x.size()[0],256,1,-1)   #[*,*,1*1]
        out_2 = out_2.reshape(x.size()[0],256,1,-1)   #[*,*,2*2]
        out_3 = out_3.reshape(x.size()[0],256,1,-1)   #[*,*,3*3]
        out_6 = out_6.reshape(x.size()[0],256,1,-1)   #[*,*,6*6]

        # 维度拼接 [*,256,1+4+9+36]  得到聚集中心F
        F = torch.cat((out_1,out_2,out_3,out_6),dim = -1)  #[*,256,1,1+4+9+36]
        C = Fun.relu(self.F_to_C(F)).reshape(x.size()[0],1,-1)     # [*,1,1+4+9+36]
        D = Fun.relu(self.F_to_D(F)).reshape(x.size()[0],256,-1).unsqueeze(-1)     # [*,256,1+4+9+36]
        B = Fun.relu(self.Input_to_B(x)).reshape(x.size()[0],1,-1) # [*,1,14*14]

        
        S = torch.zeros(x.size()[0],B.size()[-1],C.size()[-1]).to(x.device)
        
        for idx in range(x.size()[0]):
            b_reshaped = B[idx].view(B.size()[-1])
            c_reshaped = C[idx].view(C.size()[-1])
            S[idx] = torch.mm(b_reshaped.unsqueeze(1), c_reshaped.unsqueeze(0)).softmax(dim=1)
        S = S.unsqueeze(1).expand(x.size()[0],256,S.size()[1],S.size()[2])
        E = torch.matmul(S,D).reshape(x.size()[0],256,x.size()[2],x.size()[3])
        E= self.alpha(E)+x
        return E

#  通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, input,output = 50):
        super(ChannelAttention, self).__init__()

        self.Input_to_F   = nn.Conv2d(input, output, 1, bias=False)  # 使用1*1的卷积核降低通道维数

        self.beta   = nn.Conv2d(input, input, 1, bias=False)  # 使用1*1的卷积核降低通道维数
        self.output = output
        

    def forward(self, x):

        F = self.Input_to_F(x).reshape(x.size()[0],self.output,-1) # [*,256,14,14]->[*,50,14,14]->[*,50,196]
        A = x.reshape(x.size()[0],x.size()[1],-1).transpose(2,1)   # [*,196,256]
        S = torch.matmul(F,A).softmax(dim=2).transpose(2,1)   # # [*,256,50]
        E = torch.matmul(S,F).reshape(x.size()[0],x.size()[1],x.size()[2],-1)
        E = self.beta(E)+x
        return E

#  空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self,input):
        super(SpatialAttention, self).__init__()

        # 四个不同尺度卷积层
        self.conv_1 = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.conv_3 = nn.Conv2d(2, 1, kernel_size=3,padding = 1, bias=False)
        self.conv_5 = nn.Conv2d(2, 1, kernel_size=5,padding = 2, bias=False)
        self.conv_7 = nn.Conv2d(2, 1, kernel_size=7,padding = 3, bias=False)

     
        # 利用1x1卷积代替全连接
        self.A_to_B   = nn.Conv2d(input, 1, 1, bias=False)
        self.F_to_C   = nn.Conv2d(4, 1, 1, bias=False)
        self.F_to_D   = nn.Conv2d(4, input, 1, bias=False)
        self.output_to_E   = nn.Conv2d(input, input, 1, bias=False)


    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        AM = torch.cat((avg_out,max_out),dim = 1)  #[2,H,W]
        S1 = Fun.leaky_relu(self.conv_1(AM),0.001,inplace = True)
        S3 = Fun.leaky_relu(self.conv_3(AM),0.001,inplace = True)
        S5 = Fun.leaky_relu(self.conv_5(AM),0.001,inplace = True)
        S7 = Fun.leaky_relu(self.conv_7(AM),0.001,inplace = True)
        B = Fun.leaky_relu(self.A_to_B(x),0.001,inplace = True).reshape(x.size()[0],1,x.size()[2],x.size()[3])    #[1,H,W]
        F = torch.cat((S1,S3,S5,S7),dim = 1)  #[4,H,W]
        C = Fun.leaky_relu(self.F_to_C(F),0.001,inplace = True)  #[1,H,W]
        D = Fun.leaky_relu(self.F_to_D(F),0.001,inplace = True) #[256,H,W]

        S = torch.zeros(x.size()[0],B.size()[-2],C.size()[-2]).to(x.device)
        
        for idx in range(x.size()[0]):
            b_reshaped = B[idx].view([B.size()[-2],B.size()[-1]])
            c_reshaped = C[idx].view([C.size()[-1],C.size()[-2]])
            S[idx] = torch.mm(b_reshaped, c_reshaped).softmax(dim=1)
        S = S.unsqueeze(1).expand(x.size()[0],256,S.size()[1],S.size()[2])
        E = torch.matmul(S,D).reshape(x.size()[0],256,x.size()[2],x.size()[3])
        E= self.output_to_E(E)
        return E+x

# 注意力机制融合模块
class CPA_CCA_block(nn.Module):
    def __init__(self, input, output2=16):
        super(CPA_CCA_block, self).__init__()
        self.channelattention = ChannelAttention(input,output2)
        self.spatialattention = SpatialAttention(input)
        self.conv_512_to256  = nn.Conv2d(2*input, input, 1, bias=False)  # 使用1*1的卷积核降低通道维数
        self.conv_512_to256_1  = nn.Conv2d(2*input, input, 1, bias=False)  # 使用1*1的卷积核降低通道维数
        self.conv_512_to256_2  = nn.Conv2d(2*input, input, 1, bias=False)  # 使用1*1的卷积核降低通道维数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.channelattention(x)
        out2 = self.spatialattention(x)
        out  = torch.cat((out1,out2),dim = 1)  #[*,512,14,14]
        H = Fun.leaky_relu(self.conv_512_to256(out),0.001,inplace = True) #[*,256,14,14]
        M  = self.sigmoid(self.conv_512_to256_1(torch.cat((H,x),dim = 1)))
        H_M  = Fun.leaky_relu(self.conv_512_to256_2(torch.cat((H,M),dim = 1)),0.001,inplace = True)  #[*,256,14,14]
        return H_M

# # 注意力机制融合模块  OLD
# class CPA_CCA_block(nn.Module):
#     def __init__(self, input, output1=1, output2=50):
#         super(CPA_CCA_block, self).__init__()
#         self.channelattention = PositionAttention(input,output1)
#         self.spatialattention = ChannelAttention(input,output2)
#         self.conv_512_to256  = nn.Conv2d(2*input, input, 1, bias=False)  # 使用1*1的卷积核降低通道维数
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         out1 = self.channelattention(x)
#         out2 = self.spatialattention(x)
#         out  = torch.cat((out1,out2),dim = 1)  #[*,512,14,14]
#         H = self.conv_512_to256(out)           #[*,256,14,14]
#         M  = self.sigmoid(self.conv_512_to256(torch.cat((H,x),dim = 1)))
#         M = M*x
#         H_M  = self.conv_512_to256(torch.cat((H,M),dim = 1))  #[*,256,14,14]
#         return H_M
