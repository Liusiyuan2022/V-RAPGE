import torch
import torch.nn.functional as F
from PIL import Image
import os
def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps


@torch.no_grad()
def encode(model, tokenizer, text_or_image_list):
    if (isinstance(text_or_image_list[0], str)):
        inputs = {
            "text": text_or_image_list,
            'image': [None] * len(text_or_image_list),
            'tokenizer': tokenizer
        }
    else:
        inputs = {
            "text": [''] * len(text_or_image_list),
            'image': text_or_image_list,
            'tokenizer': tokenizer
        }
    outputs = model(**inputs)
    attention_mask = outputs.attention_mask
    hidden = outputs.last_hidden_state

    reps = weighted_mean_pooling(hidden, attention_mask)   
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return embeddings


def all_path_to_one_create(image_paths, max_width=3000):
    """
    将多张图片水平拼接成一张图片
    
    参数:
        image_paths: 图片路径列表
        max_width: 拼接图片的最大宽度
    
    返回:
        拼接后的图片对象
    """
    if len(image_paths) == 1:
        return Image.open(image_paths[0])
        
    images = [Image.open(img_path) for img_path in image_paths]
    
    # 确定输出图像的高度 (使用最高图片的高度)
    max_height = max(img.height for img in images)
    
    # 计算总宽度并确保不超过最大宽度
    total_width = sum(img.width for img in images)
    
    # 如果总宽度太大，按比例缩小所有图片
    if total_width > max_width:
        scale_factor = max_width / total_width
        scaled_images = []
        for img in images:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            scaled_images.append(img.resize((new_width, new_height)))
        images = scaled_images
        max_height = max(img.height for img in images)
        total_width = sum(img.width for img in images)
    
    # 创建一个新的空白图像
    concatenated_img = Image.new('RGB', (total_width, max_height))
    
    # 将所有图像粘贴到新图像中
    x_offset = 0
    for img in images:
        concatenated_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    concatenated_img_path_tmp = "./tmp/combined_image.jpg"
    
    concatenated_img.save(concatenated_img_path_tmp)
    
    return concatenated_img

def all_path_to_one_remove():
    # 清理临时文件
    if  os.path.exists("./tmp/combined_image.jpg"):
        try:
            os.remove("./tmp/combined_image.jpg")
        except:
            pass