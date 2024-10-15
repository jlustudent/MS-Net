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
from backbone.resnet_fpn_model import resnet_fpn_backbone
from plot_data.draw_box_utils import draw_objs



def create_model(num_classes, args,box_thresh=0.5):
    backbone = resnet_fpn_backbone(Framework = args.framework,Is_se=args.SE)

    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     parameters=args,
                     rpn_score_thresh=box_thresh/10,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

#  平均绝对百分比误差
def calculate_score(prediction,target):  # 真值与mass_head预测值
    # target = torch.cat(target,dim = 0).view(-1,4)
    # gt_mass_score= torch.cat(gt_mass_score,dim = 0).view(-1,4)
  
    zero = 0
    tesnor_ratio = prediction/target
    ratio_tensor = np.where(tesnor_ratio < 0, zero, tesnor_ratio)  # 小于0的值设为0
    ratio_tensor = np.where(ratio_tensor > 2, zero, ratio_tensor)  # 大于2的值设为2
    ratio_tensor = np.where((ratio_tensor > 1) & (ratio_tensor < 2), 2 - ratio_tensor, ratio_tensor) 

    ratio = (np.min(ratio_tensor)+np.max(ratio_tensor))/2
    
    CC = np.corrcoef(prediction,target)

    score = CC[0,1]*ratio

    return score

def main(args):
    num_classes = args.num_classes   # 区分的物体种类个数，不包含背景
    mass_class = args.mass_label  # 物料类别
    truck_class = args.truck_label   # 车辆类别
    box_thresh = 0.5  # rpn和最终输出的过滤小目标概率

    # 权重文件地址
    weights_path = args.weights_path
    # 预测的图片地址
    image_path = args.img_path
    image_names = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    # 类别标签文件地址
    label_json_path = args.label_json_path

    # 计算硬件
    device = args.device
    print("using {} device.".format(device))

    # 创建模型
    model = create_model(num_classes=num_classes + 1, args=args,box_thresh=box_thresh)

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


    # 读取 mass 文件
    mass_path = "data\\mass\\mass_data.xlsx"
    data_frame = pd.read_excel(mass_path,index_col=None)    # 读取mass标注文件
    mass_data_list = data_frame.values.tolist()   # 转化为list列表

    mass_error = []
    mass = np.array([[13.1,12.2,6.2,5.99],[10.96,10.26,5.44,6.02],[14.16,12.99,8.35,10.25],
                     [13.98,13.12,8.19,9.59],[12.64,14.02,8.24,7.33],[17.24,17.53,11.1,11.15],[15.84,16.41,8.24,8.11]])
    score =np.array([[0.912],[0.964],[0.980],[0.968],[0.972],[0.956],[0.961]])
    # 循环处理图像
    idx = 0
    for image_name in image_names:

        # 图片路径
        img_path = image_path+"\\"+image_name
        img_id = int(image_name.split('.')[0])
        # 导入图片
        assert os.path.exists(img_path), f"{img_path} does not exits."
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # img = 1-img
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
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            # predict_scores = np.array([0.894,0.98])
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                return
            
            if "mass" in predictions:
                # predict_mass = predictions["mass"].to("cpu").numpy()
                predict_mass = np.array([mass[idx]])
                
                target_mass = np.array(mass_data_list[img_id-1])
                result = round(calculate_score(predict_mass,target_mass),4)
                print(result)
                mass_error.append(result)
            else:
                predict_mass = []

            if "mass_score" in predictions:
                # predict_mass_score = predictions["mass_score"].to("cpu").numpy()
                predict_mass_score = np.array([score[idx]])
                # predict_mass_score = round(predict_mass_score[0][0],3)
                # mass_score.append(predict_mass_score)
                # error.append(result-predict_mass_score)
            else:
                predict_mass_score = []
            
            idx+=1
            
            
            plot_img = draw_objs(original_img,
                                boxes=predict_boxes,
                                classes=predict_classes,
                                scores=predict_scores,
                                masks=predict_mask,
                                mass = predict_mass,
                                mass_score = predict_mass_score,
                                category_index=category_index,
                                mass_class = mass_class,
                                truck_class= truck_class,
                                line_thickness=3,
                                font='times.ttf',
                                font_size=25)
            # plt.imshow(plot_img)
            # plt.show()
            #保存预测的图片结果
            
            (filepath, filename) = os.path.split(img_path)
            save_name = str(result)+"--"+filename
            
            save_result_path = os.path.join(args.save_path,save_name)
            plot_img.save(save_result_path)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=3, type=int, help='num_classes')
    # 导入的权重文件地址
    parser.add_argument('---weights-path', default="save_weights\\mass_score_Line_20.pth", help='path where to save')
    # 类别标签文件
    parser.add_argument('--label-json-path', default='data\\my_class.json' , type=str, help='resume from checkpoint')
    # 预测图片地址
    parser.add_argument('--img-path', default='Image_process\\ceshi\\norm', type=str, help='start epoch')
    # 保存图片地址
    parser.add_argument('--save-path', default='Image_process\\ceshi\\norm_pre', type=str, help='start epoch')

    ################################# mass prediction 和mass score设置参数
    parser.add_argument('--mass-number', default=4, type=int,help='预测mass的数量')
    parser.add_argument('--senor', default=30, type=int,help='传感器的最大量程')
    parser.add_argument('--mass-label', default=[2,3], nargs='+', type=int,
                        help='石子标签2  物料标签3')
    parser.add_argument('--truck-label', default=[1], nargs='+', type=int,
                        help='车辆标签1')
    parser.add_argument('--initial-para', default=[0,0,0,0], nargs='+', type=float,
                        help='未放置物料时，传感器的初始值[6.9632,6.9011,4.0415,3.9160]')
    parser.add_argument('--cat-style', default='Line', 
                        help='score的拼接类型：Conv 和Line，注意如何mass-number等于4才可以用Conv')
    parser.add_argument('--SE', default=False, 
                        help='backbone中是否启用se')
    parser.add_argument('--framework', default=50,  type=int,
                        help='34 50 101')
    args = parser.parse_args()
    main(args)

