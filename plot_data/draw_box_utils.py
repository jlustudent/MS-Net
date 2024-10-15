from PIL.Image import Image, fromarray
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import numpy as np

# 颜色库 
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              i:int,
              category_index: dict,
              color: str,
              mass_class:list = None,
              font_type: str = 'arial.ttf',
              font_size: int = 24,
              mass:list = None,
              mass_score:list = None,
              MASS_colors:str = None,
              score_colors:str = None):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        font = ImageFont.truetype(font_type, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    left_mass = left
    
    if mass == None:
        display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
        display_str_heights = [font.getsize(ds)[1] for ds in display_str]
        # Each display_str has a top and bottom margin of 0.05x.
        display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

        if top > display_str_height:
            text_top = top - display_str_height
            text_bottom = top
        else:
            text_top = top 
            text_bottom = top+ display_str_height
        

        for ds in display_str:
            text_width, text_height = font.getsize(ds)
            margin = np.ceil(0.0 * text_width)
            draw.rectangle([(left, text_top),
                            (left + text_width + 2 * margin, text_bottom)], fill=color)
            draw.text((left + margin, text_top),
                    ds,
                    fill='black',
                    font=font)
            left += text_width

    elif mass_score == None:
        # 如果将display字符串添加到边界框顶部的总高度超出了图像的顶部，
        # 那么应该将这些字符串堆叠在边界框的下方而不是上方。
        if cls in mass_class:
            display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
            mass = [round(x, 2) for x in mass[i]]
            mass_str = f"MASS(kg): {mass}"
            mass_str_heights = [font.getsize(ms)[1] for ms in mass_str]
        else:
            display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
            mass_str_heights = [0]

        display_str_heights = [font.getsize(ds)[1] for ds in display_str]
        
        # 每个display_str都有上下边距各为文本宽度的5%。.
        display_str_height = (1 + 2 * 0.09) * max(display_str_heights)
        mass_str_height = (1 + 2 * 0.09) * max(mass_str_heights)
        if top > display_str_height:
            text_top = top - display_str_height
            text_bottom = top
        else:
            text_top = top
            text_bottom = top + display_str_height

        if cls in mass_class:
            if top > mass_str_height:
                text_top = top - mass_str_height
                text_bottom = top
            else:
                text_top = top+display_str_height
                text_bottom = top + display_str_height+mass_str_height
            for ds in display_str:
                text_width, text_height = font.getsize(ds)
                margin = np.ceil(0.0 * text_width)
                draw.rectangle([(left, top),
                                (left + text_width + 2 * margin, top+display_str_height)], fill=color)
                draw.text((left + margin, top),
                        ds,
                        fill='black',
                        font=font)
                left += text_width

            for ms in mass_str:
                
                mass_width, mass_height = font.getsize(ms)
                mass_margin = np.ceil(0.0 * mass_width)
        
                # 绘制mass 信息
                draw.rectangle([(left_mass, text_top),
                                (left_mass + mass_width + 2 * mass_margin, text_bottom)], fill=MASS_colors[0])
                draw.text((left_mass + mass_margin, text_top),
                        ms,
                        fill='red',
                        font=font)
                left_mass += mass_width
        else:
            for ds in display_str:
                text_width, text_height = font.getsize(ds)
                margin = np.ceil(0.0 * text_width)
                draw.rectangle([(left, text_top),
                                (left + text_width + 2 * margin, text_top+display_str_height)], fill=color)
                draw.text((left + margin*0, text_top),
                        ds,
                        fill='black',
                        font=font)
                left += text_width

    else:
        # 如果将display字符串添加到边界框顶部的总高度超出了图像的顶部，
        # 那么应该将这些字符串堆叠在边界框的下方而不是上方。
        if cls in mass_class:
            display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
            massscore = int(1000*(round(mass_score[i][0],3)))/10
            score_str = f"Mass Score: {massscore}%"
            score_str_heights = [font.getsize(s)[1] for s in score_str]
            mass = [round(x, 2) for x in mass[i]]
            mass_str = f"MASS(kg): {mass}"
            mass_str_heights = [font.getsize(ms)[1] for ms in mass_str]
        else:
            display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
            mass_str_heights = [0]
            score_str_heights = [0]

        display_str_heights = [font.getsize(ds)[1] for ds in display_str]
        
        # 每个display_str都有上下边距各为文本宽度的5%。.
        max_height = max(score_str_heights) if max(display_str_heights)<max(score_str_heights) else max(display_str_heights)
        display_str_height = (1 + 2 * 0.09) * max_height
        mass_str_height = (1 + 2 * 0.09) * max(mass_str_heights)
        if top > display_str_height:
            text_top = top - display_str_height
            text_bottom = top
        else:
            text_top = top
            text_bottom = top + display_str_height

        if cls in mass_class:
            if top > mass_str_height:
                text_top = top - mass_str_height
                text_bottom = top
            else:
                text_top = top+display_str_height
                text_bottom = top + display_str_height+mass_str_height
            for ds in display_str:
                text_width, text_height = font.getsize(ds)
                margin = np.ceil(0.0 * text_width)
                draw.rectangle([(left, top),
                                (left + text_width + 2 * margin, top+display_str_height)], fill=color)
                draw.text((left + margin, top),
                        ds,
                        fill='black',
                        font=font)
                left += text_width

            # mass score 信息
            for s in score_str:
                
                score_width, mass_height = font.getsize(s)
                score_margin = np.ceil(0.0 * score_width)
        
                # 绘制mass 信息
                draw.rectangle([(left, top),
                                (left + score_width + 2 * score_margin,  top+display_str_height)], fill=score_colors[0])
                draw.text((left + score_margin, top),
                        s,
                        fill='blue',
                        font=font)
                left += score_width

            # mass 信息
            for ms in mass_str:
                
                mass_width, mass_height = font.getsize(ms)
                mass_margin = np.ceil(0.0 * mass_width)
        
                # 绘制mass 信息
                draw.rectangle([(left_mass, text_top),
                                (left_mass + mass_width + 2 * mass_margin, text_bottom)], fill=MASS_colors[0])
                draw.text((left_mass + mass_margin, text_top),
                        ms,
                        fill='red',
                        font=font)
                left_mass += mass_width
        else:
            for ds in display_str:
                text_width, text_height = font.getsize(ds)
                margin = np.ceil(0.0 * text_width)
                draw.rectangle([(left, text_top),
                                (left + text_width + 2 * margin, text_top+display_str_height)], fill=color)
                draw.text((left + margin*0, text_top),
                        ds,
                        fill='black',
                        font=font)
                left += text_width


def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))


def draw_objs(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              mass: np.ndarray = None,
              mass_score: np.ndarray = None,
              category_index: dict = None,
              mass_class:list = None,
              truck_class:list = None,
              box_thresh: float = 0.3,
              mask_thresh: float = 0.4,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = True):
    """
    将目标边界框信息，类别信息，mask信息绘制在图片上
    Args:
        image: 需要绘制的图片
        boxes: 目标边界框信息
        classes: 目标类别信息
        scores: 目标概率信息
        masks: 目标mask信息
        category_index: 类别与名称字典
        box_thresh: 过滤的概率阈值
        mask_thresh:
        line_thickness: 边界框宽度
        font: 字体类型
        font_size: 字体大小
        draw_boxes_on_image:
        draw_masks_on_image:

    Returns:

    """

    # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    
    if len(boxes) == 0:
        return image
    
    
    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]
    MASS_colors = [ImageColor.getrgb(STANDARD_COLORS[6 % len(STANDARD_COLORS)])]
    score_colors = [ImageColor.getrgb(STANDARD_COLORS[20 % len(STANDARD_COLORS)])]
    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks(image, masks, colors, mask_thresh)

    if draw_boxes_on_image:
        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)
        i = 0  # 记录mass的数量
        for box, cls, score, color in zip(boxes, classes, scores, colors):
            left, top, right, bottom = box
            # 绘制目标边界框
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thickness, fill=color)
            # 绘制类别和概率信息
            if len(mass) == 0 and cls in truck_class:
                draw_text(draw, box.tolist(), int(cls), float(score),i, 
                      category_index, color, font_type=font, font_size=font_size)
            elif mass_score == None and len(mass) != 0 :
                draw_text(draw, box.tolist(), int(cls), float(score),i, 
                      category_index, color,mass_class, font, font_size,mass.tolist(),None,MASS_colors,score_colors)
            else:
                draw_text(draw, box.tolist(), int(cls), float(score),i, 
                      category_index, color,mass_class, font, font_size,mass.tolist(),mass_score.tolist(),MASS_colors,score_colors)
            if int(cls) in mass_class:
                i += 1  # 记录mass的数量

    return image
