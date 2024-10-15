import math
import sys
import os
import time
import random
import torch
from torch.nn import functional as F
import train_utils.distributed_utils as utils
from train_utils.coco_eval import *
from collections import defaultdict
# from .coco_eval import EvalCOCOMetric

#  平均绝对百分比误差
def smape(prediction,target):

    error = 2.0*torch.abs(prediction-target)/(torch.abs(prediction)+torch.abs(target))
    RMSE = 1-torch.sqrt(torch.mean((torch.abs(prediction-target)/torch.abs(target))**2))
    max = torch.max(error)
    min = torch.min(error)
    SMAPE = (1-(max-min)/2)*(1-torch.mean(error))
    return SMAPE.item(),RMSE.item()
#  计算Iou 
def calculate_iou(box1, box2):
  
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2g - x1g + 1) * (y2g - y1g + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_score(prediction,target,mass_prediction,gt_mass_score,pre_score):  # 真值与mass_head预测值
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
        # elif mass==0 or CC_m[0,1]<=0:
        #     s = CC[0,1]*ratio
        # else:
        #     s = CC[0,1]*0.9*ratio+CC_m[0,1]*0.1*mass
        if mass==0 or CC_m[0,1]<=0:
            s = CC[0,1]*ratio
        else:
            s = CC_m[0,1]*mass

        s = torch.clamp(s, min=0, max=1)   
        score.append(s.unsqueeze(0))

    score = torch.cat(score,dim = 0).unsqueeze(1)
    
    # mse_loss = 2*torch.abs(pre_score-score)/(torch.abs(pre_score)+torch.abs(score))
    mse_loss = torch.sqrt(torch.abs(pre_score-score)**2)
        # 计算均方根
    rmse_accu = 1-mse_loss

    return rmse_accu.item()    

#  训练函数
    # model 网络模型
    # optimizer 设置的优化器
    # train_data_loader  导入训练数据集
    # device 训练的设备（默认GPU）
    # epoch 当前的训练次数
    # print_freq每迭代50次打印一次
def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=100, warmup=False, scaler=None):
    
    # 网络训练模式
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
    # 目的;初始训练如果学习率过大会是模型不稳，因此在前几个step中使用小的学习率
    if epoch == 0 and warmup is True:  
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        # 开始使用一个较低的学习率
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 记录总的平均损失
    mloss = torch.zeros(1).to(device)
    mass_loss = torch.zeros(1).to(device)
    mass_score_loss = torch.zeros(1).to(device)
    # mask_loss = torch.zeros(1).to(device)
    class_loss = torch.zeros(1).to(device)  
    box_reg_loss = torch.zeros(1).to(device)

   

    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        target= {}   # 记录拼接后的目标信息
        first_mask = None
        second_mask= None
        mask1= None
        mask2= None
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #  ！！！！！！！！此处用于将大小相同的图像随机进行横向或纵向拼接  
        if random.random()<0.5 and len(targets)==2:
            if random.random()<0.5:   # 纵向拼接
                
                # 填充mask
                first_mask = torch.zeros_like(targets[0]["masks"])
                second_mask = torch.zeros_like(targets[1]["masks"])
                # 坐标转换
                targets[1]["boxes"][:,[1,3]] = targets[1]["boxes"][:,[1,3]]+images[0].shape[1]
                # 信息拼接
                target["boxes"] = torch.cat((targets[0]["boxes"], targets[1]["boxes"]), dim=0)
                target["labels"] = torch.cat((targets[0]["labels"], targets[1]["labels"]), dim=0)
                target["mass"] = torch.cat((targets[0]["mass"], targets[1]["mass"]), dim=0)
                target["ratio"] = torch.cat((targets[0]["ratio"], targets[1]["ratio"]), dim=0)
                # 扩充mask
                mask1 = torch.cat((targets[0]["masks"],first_mask),dim = 1)
                mask2 = torch.cat((second_mask,targets[1]["masks"]),dim = 1)
                # 拼接mask
                target["masks"] = torch.cat((mask1,mask2),dim = 0)

                images = [torch.cat((images[0], images[1]), dim=1)]
                target = [target]

            # 横向拼接  
            else:    
                # 填充mask
                first_mask = torch.zeros_like(targets[0]["masks"])
                second_mask = torch.zeros_like(targets[1]["masks"])
                # 坐标转换
                targets[1]["boxes"][:,[0,2]] = targets[1]["boxes"][:,[0,2]]+images[0].shape[2]
                # 信息拼接
                target["boxes"] = torch.cat((targets[0]["boxes"], targets[1]["boxes"]), dim=0)
                target["labels"] = torch.cat((targets[0]["labels"], targets[1]["labels"]), dim=0)
                target["mass"] = torch.cat((targets[0]["mass"], targets[1]["mass"]), dim=0)
                target["ratio"] = torch.cat((targets[0]["ratio"], targets[1]["ratio"]), dim=0)
                # 扩充mask
                mask1 = torch.cat((targets[0]["masks"],first_mask),dim = 2)
                mask2 = torch.cat((second_mask,targets[1]["masks"]),dim = 2)
                # 拼接mask
                target["masks"] = torch.cat((mask1,mask2),dim = 0)

                images = [torch.cat((images[0], images[1]), dim=2)]
                target = [target]
        else:
            target = targets


        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, target,epoch)

            # 计算总损失
            losses = sum(loss for loss in loss_dict.values())

        # 减少GPU的损失进行 日志记录
        loss_dict_reduced = utils.reduce_dict(loss_dict)   # 对于单GPU 不做任何处理  字典类型
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())  # 对所有的损失求和

        if "loss_mass" in loss_dict_reduced:
            loss_mass = loss_dict_reduced["loss_mass"].item()
        else:
            loss_mass = mass_loss 
        if "loss_mass_score" in loss_dict_reduced:
            loss_mass_score = loss_dict_reduced["loss_mass_score"].item()
        else:
            loss_mass_score = mass_score_loss 
        # loss_mask = loss_dict_reduced["loss_mask"].item()
        # loss_class = loss_dict_reduced["loss_classifier"].item()
        # loss_box = loss_dict_reduced["loss_box_reg"].item()
        loss_value = losses_reduced.item()  # tensor转化为标量

        # 记录训练平均损失   （返回）
        mass_loss = (mass_loss * i + loss_mass) / (i + 1)
        mass_score_loss = (mass_score_loss * i + loss_mass_score) / (i + 1)
        # mask_loss = (mask_loss * i + loss_mask) / (i + 1) 
        # class_loss = (class_loss * i + loss_class) / (i + 1)
        # box_reg_loss = (box_reg_loss * i + loss_box) / (i + 1)
        mloss = (mloss * i + loss_value) / (i + 1)  

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, 趋近无限大，停止训练".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        # 优化器梯度清零
        optimizer.zero_grad()
        # scaler 混合精度对象
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 反向传播计算每个参数的梯度值
            losses.backward()
            # 参数更新
            optimizer.step()

         # 第一轮使用warmup训练方式，更新学习率
        if lr_scheduler is not None: 
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    mass_loss = round(mass_loss.item(),4)
    mass_score_loss = round(mass_score_loss.item(),4)
    # mask_loss = round(mask_loss.item(),4)
    # class_loss = round(class_loss.item(),4) 
    # box_reg_loss = round(box_reg_loss.item(),4)

    LOSS = [mass_loss,mass_score_loss]
    return LOSS


@torch.no_grad()  # 函数中的数据不需要计算梯度
def evaluate(model, data_loader, args,device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")  # 用于跟踪和记录模型训练过程中的指标，比如损失函数的值、准确率、学习率
    header = "Test: "

    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name="evaluate_results\\det_results.json")
    seg_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="segm", results_file_name="evaluate_results\\seg_results.json")

    Cycle_num = 0  # 循环次数记录
    Cycle_score = 0
    Mean_Accuracy = 0
    RMSE_Value = 0
    Mean_Score_Accuracy = 0
    fault_num = 0  # 验证集的无法识别失败的数量
    val_num = 300 # 验证集的数据长度
    score_accuracy = 0

    coco_results = []
    # 初始化用于存储每个类别的 TP、FP、FN
    class_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        # 模型预测输出
        model_time = time.time()
        outputs = model(image)

        # 模型输出的预测结果为空时，evaluate()函数会报错
        if outputs is None:
            continue
        
        # 网络预测输出的进行设备替换
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time # 预测时间

          ########################### 计算 TP FP F1 AP################################
         # 处理每个批次的输出结果
        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"].item()

            # 获取真实框和类别
            gt_boxes = targets[i]['boxes'].tolist()
            gt_labels = targets[i]['labels'].tolist()
            gt_matched = [False] * len(gt_boxes)  # 用于标记哪些 GT 已被匹配

            # 获取预测框、类别和得分
            pred_boxes = output['boxes'].tolist()
            pred_labels = output['labels'].tolist()
            pred_scores = output['scores'].tolist()

            # 过滤掉得分低于阈值的预测
            keep = [j for j, score in enumerate(pred_scores) if score >= 0]
            pred_boxes = [pred_boxes[j] for j in keep]
            pred_labels = [pred_labels[j] for j in keep]
            pred_scores = [pred_scores[j] for j in keep]
            
            # 匹配预测框和真实框
            for pred_box, pred_label in zip(pred_boxes, pred_labels):
                matched = False
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_label != pred_label or gt_matched[gt_idx]:
                        continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou and iou >= 0.5:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0:
                    # 匹配成功，计为 TP
                    class_metrics[pred_label]['TP'] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    # 没有匹配的 GT，计为 FP
                    class_metrics[pred_label]['FP'] += 1

            # 统计未匹配的 GT，计为 FN
            for gt_idx, matched in enumerate(gt_matched):
                if not matched:
                    gt_label = gt_labels[gt_idx]
                    class_metrics[gt_label]['FN'] += 1


            # 提取预测的类别、边界框、得分
            for j, box in enumerate(output['boxes']):
                score = output['scores'][j].item()
                category_id = output['labels'][j].item()
                bbox = box.cpu().numpy().tolist()

                # 将预测结果转换为 COCO 格式并保存
                coco_results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # COCO 格式
                    "score": score
                })

        # 计算各种评估指标，如Precision、Recall、AP（Average Precision）
        
        metric_logger.update(model_time=model_time)

        if "mass_score" in outputs[0]or "mass" in outputs[0]:
            initial_data = torch.tensor(args.initial_para)
            # 乘以影响因子的质量系数
            score_true = (targets[0]["mass"]-initial_data)/(args.senor-initial_data)*targets[0]["ratio"] 
            # 载荷分布数值（乘以影响因子之后的）
            gt_mass = score_true*(args.senor-initial_data)+initial_data
            # 网络预测的质量系数
            mass_pre = outputs[0]["weights"]
            if score_true.shape[0]!=outputs[0]["mass_score"].shape[0]:
                print("数据长度不一致")
                # os.system("pause")
                score_accuracy = 0
                Accuracy = 0
                fault_num+=1
                
            else:
                score_accuracy = calculate_score(mass_pre,score_true,outputs[0]["mass"],gt_mass,outputs[0]["mass_score"])
                Accuracy,RMSE_accu = smape(outputs[0]["mass"],gt_mass)
        else:
            score_accuracy = 0
            Cycle_score-=1
            Accuracy = 0
            RMSE_accu = 0
            Cycle_num-=1
            fault_num+=1


        Mean_Accuracy+=Accuracy
        Mean_Score_Accuracy += score_accuracy
        RMSE_Value+=RMSE_accu
        Cycle_num +=1
        Cycle_score +=1 

    # 计算平均精确度
   
    Mean_Accuracy = round(Mean_Accuracy/Cycle_num*((val_num-fault_num)/val_num),4)
    RMSE_Value = round(RMSE_Value/Cycle_num*((val_num-fault_num)/val_num),4)
    Mean_Score_Accuracy = round(Mean_Score_Accuracy/Cycle_score*((val_num-fault_num)/val_num),4)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)

    # 计算每个类别的精确率、召回率和 F1 分数
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    for cls, metrics in class_metrics.items():
        TP = metrics['TP']
        FP = metrics['FP']
        FN = metrics['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precision_dict[cls] = precision
        recall_dict[cls] = recall
        f1_dict[cls] = f1

    # 计算宏平均指标
    num_classes = len(precision_dict)
    if num_classes > 0:
        macro_precision = sum(precision_dict.values()) / num_classes
        macro_recall = sum(recall_dict.values()) / num_classes
        macro_f1 = sum(f1_dict.values()) / num_classes
    else:
        macro_precision, macro_recall, macro_f1 = 0, 0, 0

 
    ACCURACY = [Mean_Accuracy,RMSE_Value,Mean_Score_Accuracy,macro_precision, macro_recall, macro_f1]

    return ACCURACY


# def calculate_score(prediction,target,pre_score):  # 真值与mass_head预测值

#     # target = torch.cat(target,dim = 0).view(-1,4)   # [all_mass_proposal,4]
#     device = prediction.device
#     target = target.to(device) # 修改变量的设备类型
#     # 网络输出的允许梯度传递，标签数据禁止梯度传递
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

#     max = torch.max(mass_error, dim=1, keepdim=True)[0]
#     min = torch.min(mass_error, dim=1, keepdim=True)[0]
#     average_percent = torch.mean(mass_percent, dim=1, keepdim=True)
#     score = (1-(max-min)/2)*average_percent
#     is_close_to_zero = torch.allclose(mass_percent, torch.zeros_like(mass_percent))
#     if is_close_to_zero:
#         raise ValueError("计算分数为全零")
#     else:
#         mse_loss = F.mse_loss(score, pre_score)
#         # 计算均方根
#         rmse_accu = 1-torch.sqrt(mse_loss)
#     return rmse_accu.item()



# def calculate_score(prediction,target,mass_pre,mass_true,pre_score):  # 真值与mass_head预测值

#     R2 = R_squared(prediction,target)
#     dr = DR(mass_pre,mass_true)
#     if R2 == None and dr == None:
#         score = None
#         print("R2与DR出现分母为零的情况")
#         return score
#     elif R2 == None:
#         score = dr
#         mse_loss = F.mse_loss(score, pre_score)
#         # 计算均方根
#         rmse_accu = 1-torch.sqrt(mse_loss)
#     elif dr == None:
#         score = R2
#         mse_loss = F.mse_loss(score, pre_score)
#         # 计算均方根
#         rmse_accu = 1-torch.sqrt(mse_loss)
#     else:
#         score = 0.6*R2+0.4*dr
#         mse_loss = F.mse_loss(score, pre_score)
#         # 计算均方根
#         rmse_accu = 1-torch.sqrt(mse_loss)
#     return rmse_accu.item()