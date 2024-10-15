import os
import time
import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files.mask_rcnn import MaskRCNN
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from plot_data.draw_box_utils import draw_objs



def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()

    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

#  平均绝对百分比误差
def calculate_score(prediction,target):  # 真值与mass_head预测值
    # target = torch.cat(target,dim = 0).view(-1,4)
    # gt_mass_score= torch.cat(gt_mass_score,dim = 0).view(-1,4)
    device = prediction.device
    zero = torch.tensor(0.).to(device)
    score = []
 
    input  = torch.cat((prediction.unsqueeze(0),target.unsqueeze(0)),dim = 0)
    

    tesnor_ratio = prediction.unsqueeze(0)/target.unsqueeze(0)
    ratio_tensor = torch.where(tesnor_ratio < 0, zero, tesnor_ratio)  # 小于0的值设为0
    ratio_tensor = torch.where(ratio_tensor > 2, zero, ratio_tensor)  # 大于2的值设为2
    ratio_tensor = torch.where((ratio_tensor > 1) & (ratio_tensor < 2), 2 - ratio_tensor, ratio_tensor) 

    ratio_min = torch.min(ratio_tensor)

    ratio_max = torch.max(ratio_tensor)


    ratio = torch.mean(ratio_tensor)
    
    CC = torch.corrcoef(input)

    score = CC[0,1]*ratio

    return score
def main():
    num_classes = 3   # 区分的物体种类个数，不包含背景
    mass_class = [2,3]
    box_thresh = 0.3  # rpn和最终输出的过滤小目标概率

    # 类别标签文件地址
    label_json_path = 'data\\my_class.json'

    network_id = "AM2"
    val_path = "val_result"
    mass_path = "data\\mass\\mass_data.xlsx"
    data_frame = pd.read_excel(mass_path,index_col=None)    # 读取mass标注文件
    mass_data_list = data_frame.values.tolist()   # 转化为list列表
    # 权重文件地址
    weights_path = "./save_weights/model_attention_mask_conv2.pth"
    #weights_path = "pre_training_weights\\maskrcnn_resnet50_fpn_coco.pth"
    

    # 计算硬件
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建模型
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # 导入权重文件
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # 读取类别标签文件
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)


   
    folder_path = 'mult'

    # 获取文件夹中所有文件的名称
    image_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    mass_data = []
    
    # 打印图像名称
    for image_name in image_names:
        # 预测的图片地址
        img_path = folder_path+"\\"+image_name
        img_id = int(image_name.split('.')[0])
        
        # 导入图片
        assert os.path.exists(img_path), f"{img_path} does not exits."
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            # print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                return
            
            if "mass" in predictions:
                predict_mass = predictions["mass"].to("cpu").numpy()
                
                target_mass = np.array(mass_data_list[img_id-1])
                # error = np.round((predict_mass-target_mass)/target_mass,3)*100
                # concatenated_array = np.round(np.concatenate([predict_mass[0], target_mass]),3)
                # data = np.round(np.concatenate([concatenated_array, error[0]]),3)
                # # 保存预测值与真实值
                # mass_data.append(data)
                

                # if img_id == 22328:
                #     predict_mass = 0.96*predict_mass
                # else:
                #     predict_mass = 0.97*predict_mass
                result = round(calculate_score(predict_mass,target_mass),4)
                
                print(image_name,result)
            else:
                predict_mass = []



    # 保存所有数据
    DATA = []
    DATA.extend(mass_data)
  # 创建DataFrame
    mass_new = pd.DataFrame(DATA)

    # 读取现有的Excel文件
    mass_xlsx = pd.read_excel("attention_mask_conv_data.xlsx")

    # 将新的DataFrame追加到现有DataFrame中
    mass_updated = pd.concat([mass_xlsx, mass_new], ignore_index=True)

    # 将更新后的DataFrame写入到Excel文件中，不包括索引
    mass_updated.to_excel("attention_mask_conv_data.xlsx", index=False)


if __name__ == '__main__':
    main()
    # p = np.array([9.65,10.1,5.37,5.53])



    # t =  np.array([10.135,11.066,6.138,5.47])
    # a = accur_smape(p,t)
    # print(a)

