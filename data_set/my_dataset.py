import os
import json
import pandas as pd
import torch
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from  data_set.coco_utils import convert_coco_poly_mask

class CocoDetection(data.Dataset):
    """`

    参数:
        root (string): 为labelmeb标签转化为coco格式后，文件的的保存路径（data\\coco）.
        dataset (string): train or val. 训练模式还是验证模式
        transforms (callable, optional): 对导入的图片进行类型转化与随机翻转.
    """

    def __init__(self, root, dataset="train", transforms=None):
        super(CocoDetection, self).__init__()

        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'

        #   COCO数据集

        # anno_file = f"instances_{dataset}2017.json"

        # assert os.path.exists(root), "file '{}' does not exist.".format(root)

        # self.img_root = os.path.join(root,f"{dataset}2017")  #coco数据的文件夹地址

        # self.anno_path = os.path.join(root, "annotations", anno_file) # coco数据文件夹中的标注文件地址annotations.json

        # assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)


        #   矿卡数据集    root  = ".\data\\mass"
        mass_json = f"mass_{dataset}.json"    #矿卡图片的标注信息文件

        assert os.path.exists(root), "file '{}' does not exist.".format(root)

        self.img_root = os.path.join(root,f"{dataset}")  # 矿卡数据集训练（验证）图片的文件夹地址

        self.anno_path = os.path.join(root, "annotations" ,mass_json)  #质量标注文件路径
        
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.mass_xlsx_path = os.path.join(root,"mass_data.xlsx") # 保存图片的质量信息的excel文件路径

        data_frame = pd.read_excel(self.mass_xlsx_path,index_col=None)    # 读取mass标注文件
        self.mass_data_list = data_frame.values.tolist()   # 转化为list列表

        ###########  读取复杂图像的评分系数 ##################
        self.complex_path = os.path.join(self.img_root,f"{dataset}"+"_data.xlsx")
        complex_data = pd.read_excel(self.complex_path)    # 读取复杂系数标注文件标注文件
        self.score_ratio_dict = dict(zip(complex_data["Image Name"],complex_data["Value"]))   # 转化为字典格式列表
        
        self.mode = dataset            # 输入模式：训练模式或者验证模式
        self.transforms = transforms   # 图片的转换方式
        self.coco = COCO(self.anno_path)

        # 获取coco数据索引与类别名称的关系
        # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        max_index = max(data_classes.keys())  # 90
        # 将缺失的类别名称设置成N/A
        coco_classes = {}
        for k in range(1, max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = "N/A"

        if dataset == "train":
            json_str = json.dumps(coco_classes, indent=4)
            with open("data\\my_class.json", "w") as f:
                f.write(json_str)

        self.coco_classes = coco_classes

        ids = list(sorted(self.coco.imgs.keys()))   # 按照图片的数量的数组  【0,1,2,3,4,5,6.。。。。。】
        # if dataset == "train":
        #     # 移除没有目标，或者目标面积非常小的数据
        #     valid_ids = coco_remove_images_without_annotations(self.coco, ids)
        #     self.ids = valid_ids
        # else:
        self.ids = ids

    # 对于图片的标注信息进行转换
    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None,
                      img_name = None):
        
        # img_id为图片的索引号
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # 转化为tensor形式
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_mask(segmentations, h, w)

        # 筛选出合法的目标，即x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        #  mass标签数据处理
        mass_true = self.mass_data_list[img_name-1] # 之所以减1是因为 图片的索引是从1开始的  list索引是从0开始的
        mass_true = [x for i, x in enumerate(mass_true)]
        mass_true = torch.as_tensor(mass_true, dtype=torch.float32).reshape(-1, 4)
        target["mass"] = mass_true

        #  score系数标签数据处理
        score_img = f"{img_name}"+".jpg"
        # byte_string = score_img.encode('utf-8')
        # # 使用PyTorch创建一个张量
        # image_name = torch.tensor(byte_string, dtype=torch.uint8)
        target["image_name"] = torch.tensor([img_name])
        if score_img in self.score_ratio_dict.keys():
            score_true = [self.score_ratio_dict[score_img]]        # 若图片的名称在score_ratio文件中则取出对应的系数
            score_true = torch.as_tensor(score_true, dtype=torch.float32)
            target["ratio"] = score_true
        else:
            score_true = torch.as_tensor([1.0], dtype=torch.float32)
            target["ratio"] = score_true

        return target

    # 重写dataset类的getitem函数
    def __getitem__(self, index):
        """
        参数:
            index (int): 图片的索引序号

        返回值:
            tuple: Tuple (image, target). 返回经过transforms转化之后的图片和标注信息
            其中，target中增加了mask标注元素.
        """
        coco = self.coco
        img_id = self.ids[index]   # 图片的索引(并不是图片的名称)
        ann_ids = coco.getAnnIds(imgIds=img_id)  # 图片中标注的信息的索引
        coco_target = coco.loadAnns(ann_ids)  
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB') 
        if "JPEGImages" in  path:
            img_name = int(path.split('\\')[-1].split('.')[0])

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h,img_name)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

            # 对图像进行取反操作
            # img = 1-img
        
        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    train = CocoDetection("data\\coco", dataset="val")
    print(len(train))
    t = train[0]
