import random
import colorsys
import itertools


def generate_high_contrast_colors(num_colors):
    """
    生成num_colors个色差大的颜色（重点：打乱色相顺序，保证相邻色差大）
    Args:
        num_colors: 颜色数量
    Returns:
        colors: 颜色列表
    """
    colors = []
    saturation = 1.0  # 固定饱和度为1，保证颜色鲜艳
    value = 1.0       # 固定明度为1，保证亮度一致
    hue_step = 1.0 / num_colors
    
    # 第一步：生成均匀分布的色相列表（0-1）
    hue_list = [i * hue_step for i in range(num_colors)]
    # 第二步：随机洗牌色相列表，让相邻色相差异最大化
    random.shuffle(hue_list)
    
    # 转换为RGB颜色
    for hue in hue_list:
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        colors.append((r, g, b))
    return colors


def get_pair_combinations(str_list):
    """
    对字符串列表生成两两不重复的组合对
    
    参数:
        str_list: 输入的字符串列表，如 ['a','b','c']
    返回:
        包含两两组合元组的列表，如 [('a','b'),('a','c'),('b','c')]
    """
    # 使用combinations生成长度为2的组合，再转换为列表
    return list(itertools.combinations(str_list, 2))

