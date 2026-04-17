import numpy as np
import random
import cv2
import os
from loguru import logger
from typing import Tuple, Optional


def get_z_values_set(free_position: np.ndarray) -> np.ndarray:
    """
    获取所有唯一的z值集合
    
    参数:
        free_position: free positions数组,形状为 (n, 4),包含 [x, y, z, semantic_label] 坐标
    
    返回:
        z_values: 排序后的唯一z值数组
    """
    z_values = np.unique(free_position[:, 2])
    z_values = np.sort(z_values)
    return z_values

def filter_free_positions_at_z(free_position: np.ndarray, z_value: float, tolerance: float = 0.01) -> np.ndarray:
    """
    在指定z高度上筛选出所有free positions
    
    参数:
        free_position: free positions数组,形状为 (n, 4),包含 [x, y, z, semantic_label] 坐标
        z_value: 目标z高度
        tolerance: z值的容差(用于浮点数比较）
    
    返回:
        positions_xy: 形状为 (n, 2) 的数组,包含 [x, y] 坐标
    """
    # 筛选出指定z高度附近的点(考虑浮点数精度）
    mask = np.abs(free_position[:, 2] - z_value) < tolerance
    positions_xy = free_position[mask][:, :2]  # 只保留 x, y 坐标
    return positions_xy

def check_point_density_uniformity(
    point: np.ndarray,
    positions_xy: np.ndarray,
    radius: float = 0.5,
    num_sectors: int = 8,
    min_points_per_sector: int = 3,
    precomputed_distances: Optional[np.ndarray] = None
) -> Tuple[bool, int]:
    """
    检查某个点周围是否有均匀分布的点
    
    参数:
        point: 要检查的点, 形状为 (2,), 包含 [x, y] 坐标
        positions_xy: 所有可用的位置点数组, 形状为 (n, 2)
        radius: 检查的半径
        num_sectors: 将圆划分为多少个扇形区域
        min_points_per_sector: 每个扇形区域内至少需要多少个点
        precomputed_distances: 预计算的距离数组（可选，用于性能优化）
    
    返回:
        is_uniform: 是否均匀分布
        total_points: 圆内总点数
    """
    # 使用预计算的距离或计算新的距离
    if precomputed_distances is not None:
        distances = precomputed_distances
    else:
        # 使用更快的距离计算方法
        diff = positions_xy - point
        distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # 筛选出半径范围内的点
    within_radius_mask = distances <= radius
    total_points = np.sum(within_radius_mask)
    
    # 如果圆内点数太少, 直接返回False
    if total_points < num_sectors * min_points_per_sector:
        return False, total_points
    
    # 只对半径内的点计算角度
    points_within = positions_xy[within_radius_mask]
    dx = points_within[:, 0] - point[0]
    dy = points_within[:, 1] - point[1]
    angles = np.arctan2(dy, dx)  # 范围 [-π, π]
    angles = (angles + 2 * np.pi) % (2 * np.pi)  # 转换到 [0, 2π)
    
    # 使用向量化操作统计每个扇形区域的点数
    sector_size = 2 * np.pi / num_sectors
    sector_indices = (angles / sector_size).astype(int)
    sector_indices = np.minimum(sector_indices, num_sectors - 1)  # 处理边界情况
    
    # 使用 bincount 快速统计每个扇形的点数
    sector_counts = np.bincount(sector_indices, minlength=num_sectors)
    
    # 检查是否每个扇形区域都有足够的点
    is_uniform = np.all(sector_counts >= min_points_per_sector)
    
    return is_uniform, total_points

def is_point_near_boundary(
    point: np.ndarray,
    bounds: Tuple[float, float, float, float],
    boundary_margin: float = 0.5
) -> bool:
    """
    检查某个点是否靠近边界
    
    参数:
        point: 要检查的点, 形状为 (2,), 包含 [x, y] 坐标
        bounds: 边界元组 (x_min, y_min, x_max, y_max)
        boundary_margin: 距离边界的安全距离
    
    返回:
        True 如果点靠近边界, False 否则
    """
    x_min, y_min, x_max, y_max = bounds
    
    # 检查点是否在边界的安全距离内
    if (point[0] <= x_min + boundary_margin or 
        point[0] >= x_max - boundary_margin or
        point[1] <= y_min + boundary_margin or 
        point[1] >= y_max - boundary_margin):
        return True
    
    return False

def generate_random_path_from_positions(
    positions_xy: np.ndarray,
    num_points: int = 20,
    step_size: Optional[float] = None,
    max_angle_deviation: float = 45.0,
    check_density: bool = True,
    density_radius: float = 0.4,
    num_sectors: int = 8,
    min_points_per_sector: int = 3,
    avoid_boundary: bool = True,
    boundary_margin: float = 0.5
) -> np.ndarray:
    """
    从给定的free positions中生成随机路径, 起点和中间点都从给定的positions中选择
    
    参数:
        positions_xy: 可用的位置点数组,形状为 (n, 2),包含 [x, y] 坐标
        num_points: 路径点的数量
        step_size: 每步的最大步长(如果为None,则自动计算）
        max_angle_deviation: 最大角度偏差(度）,默认45度
        check_density: 是否检查点周围的密度均匀性
        density_radius: 检查密度的半径, 默认0.5
        num_sectors: 将圆划分为多少个扇形区域检查均匀性, 默认8
        min_points_per_sector: 每个扇形区域内至少需要多少个点, 默认3
        avoid_boundary: 是否避免选择边界附近的点, 默认True
        boundary_margin: 距离边界的安全距离, 默认0.5
    
    返回:
        path: numpy数组,形状为 (num_points, 2),包含 [x, y] 坐标
    """
    # 计算边界
    x_min, y_min = positions_xy.min(axis=0)
    x_max, y_max = positions_xy.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    bounds = (x_min, y_min, x_max, y_max)
    
    if step_size is None:
        step_size = min(width, height) * 0.1  # 自动计算每步的最大步长
    
    # 将角度偏差转换为弧度
    max_deviation_rad = np.deg2rad(max_angle_deviation)
    
    # 预先过滤掉边界附近的点，以加速后续查找
    valid_point_mask = np.ones(len(positions_xy), dtype=bool)
    if avoid_boundary:
        valid_point_mask = (
            (positions_xy[:, 0] > x_min + boundary_margin) &
            (positions_xy[:, 0] < x_max - boundary_margin) &
            (positions_xy[:, 1] > y_min + boundary_margin) &
            (positions_xy[:, 1] < y_max - boundary_margin)
        )
        # logger.info(f"边界过滤后剩余点数: {np.sum(valid_point_mask)}/{len(positions_xy)}")
    
    valid_indices = np.where(valid_point_mask)[0]
    
    path = []
    
    # 从可用位置中随机选择起始点, 如果启用密度检查, 则确保起始点周围有均匀分布的点
    max_attempts = 100
    start_idx = None
    for attempt in range(max_attempts):
        # 从有效的点中随机选择
        if len(valid_indices) > 0:
            candidate_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
        else:
            candidate_idx = random.randint(0, len(positions_xy) - 1)
        
        candidate_point = positions_xy[candidate_idx]
        
        if not check_density:
            start_idx = candidate_idx
            break
        
        # 检查候选点周围的密度均匀性
        is_uniform, total_points = check_point_density_uniformity(
            candidate_point,
            positions_xy,
            density_radius,
            num_sectors,
            min_points_per_sector
        )
        
        if is_uniform:
            start_idx = candidate_idx
            break
    
    # 如果找不到合适的起始点, 使用随机点并给出警告
    if start_idx is None:
        logger.warning(f"无法找到周围有均匀分布点且不在边界的起始点, 使用随机起始点")
        if len(valid_indices) > 0:
            start_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
        else:
            start_idx = random.randint(0, len(positions_xy) - 1)
    
    current_x, current_y = positions_xy[start_idx]
    path.append([current_x, current_y])
    
    # 初始方向随机选择
    current_angle = random.uniform(0, 2 * np.pi)
    
    for _ in range(num_points - 1):
        # 生成相对于当前方向的角度偏差(-45度到+45度之间）
        angle_deviation = random.uniform(-max_deviation_rad, max_deviation_rad)
        # 新的方向 = 当前方向 + 角度偏差
        angle = current_angle + angle_deviation
        # 归一化角度到 [0, 2π)
        angle = angle % (2 * np.pi)
        
        step = random.uniform(0.3 * step_size, step_size)
        
        # 计算下一个点的理想位置
        next_x = current_x + step * np.cos(angle)
        next_y = current_y + step * np.sin(angle)
        
        # 使用更快的距离计算方法
        diff = positions_xy - np.array([next_x, next_y])
        distances = np.sqrt(np.sum(diff**2, axis=1))
        
        # 如果启用密度检查或边界检查, 则在候选点中选择合适的点
        if check_density or avoid_boundary:
            # 找出距离理想位置较近的所有候选点
            candidate_mask = distances < step_size * 2
            
            if avoid_boundary:
                # 同时应用边界过滤
                candidate_mask = candidate_mask & valid_point_mask
            
            candidate_indices = np.where(candidate_mask)[0]
            
            if len(candidate_indices) == 0:
                # 如果没有足够近的候选点, 放宽条件
                candidate_mask = distances < step_size * 3
                if avoid_boundary:
                    candidate_mask = candidate_mask & valid_point_mask
                candidate_indices = np.where(candidate_mask)[0]
            
            # 如果仍然没有候选点，使用所有有效点
            if len(candidate_indices) == 0:
                if avoid_boundary and np.sum(valid_point_mask) > 0:
                    candidate_indices = valid_indices
                else:
                    candidate_indices = np.arange(len(positions_xy))
            
            # 在候选点中寻找周围有均匀分布点的点
            best_idx = None
            best_score = -1
            
            # 限制检查的候选点数量以提高性能
            check_limit = min(30, len(candidate_indices))  # 减少到30以提高速度
            checked_indices = np.random.choice(candidate_indices, check_limit, replace=False) if len(candidate_indices) > check_limit else candidate_indices
            
            for idx in checked_indices:
                # 检查密度均匀性
                is_uniform = True
                if check_density:
                    # 传入预计算的距离以加速
                    is_uniform, total_points = check_point_density_uniformity(
                        positions_xy[idx],
                        positions_xy,
                        density_radius,
                        num_sectors,
                        min_points_per_sector,
                        precomputed_distances=None  # 每个点的距离不同，无法重用
                    )
                
                # 如果找到均匀的点, 优先选择距离理想位置最近的
                if is_uniform:
                    score = -distances[idx]  # 距离越近分数越高
                    if score > best_score:
                        best_score = score
                        best_idx = idx
            
            # 如果找到合适的点, 使用它; 否则从候选点中选择最近的
            if best_idx is not None:
                nearest_idx = best_idx
            else:
                # 从候选点中选择最近的
                nearest_idx = candidate_indices[np.argmin(distances[candidate_indices])]
        else:
            # 不检查密度和边界, 直接使用最近的点
            nearest_idx = np.argmin(distances)
        
        nearest_point = positions_xy[nearest_idx]
        
        # 如果最近的点太远(超过步长的3倍）,则随机选择一个附近的点
        if distances[nearest_idx] > step_size * 3:
            # 选择距离当前点在一定范围内的点
            nearby_mask = distances < step_size * 3
            if np.any(nearby_mask):
                nearby_indices = np.where(nearby_mask)[0]
                nearest_idx = random.choice(nearby_indices)
                nearest_point = positions_xy[nearest_idx]
            else:
                # 如果没有附近的点,随机选择一个
                nearest_idx = random.randint(0, len(positions_xy) - 1)
                nearest_point = positions_xy[nearest_idx]
        
        next_x, next_y = nearest_point  # 更新当前位置
        
        path.append([next_x, next_y])
        
        # 更新当前位置和方向
        dx = next_x - current_x
        dy = next_y - current_y
        if dx != 0 or dy != 0:
            current_angle = np.arctan2(dy, dx)  # 更新当前方向
            # 归一化角度
            if current_angle < 0:
                current_angle += 2 * np.pi
        
        current_x, current_y = next_x, next_y
    
    return np.array(path)

def save_paths_to_npy(paths_xy: list, z_values: list, output_path: str):
    """
    将多条路径点保存为npy文件,每个路径点包含 [x, y, z] 坐标
    
    参数:
        paths_xy: 路径点数组列表,每个元素形状为 (n, 2),包含 [x, y] 坐标
        z_values: z高度值列表,长度应与paths_xy相同
        output_path: 输出文件路径
    """
    # 保存为列表(因为路径长度可能不同）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    paths_xyz = []
    for path_xy, z_value in zip(paths_xy, z_values):
        z_coords = np.full((len(path_xy), 1), z_value)
        path_xyz = np.hstack([path_xy, z_coords])
        paths_xyz.append(path_xyz)
    paths_xyz = np.array(paths_xyz)
    np.save(output_path, paths_xyz)
    return paths_xyz

def visualize_path(path_xy: np.ndarray, positions_xy: np.ndarray, z_value: float, output_path: Optional[str] = None, scale: float = 50.0, min_image_size: int = 2000):
    """
    使用OpenCV可视化路径,同时显示所有free positions作为背景点
    
    参数:
        path_xy: 路径点数组,形状为 (n, 2),包含 [x, y] 坐标
        positions_xy: 所有free positions,形状为 (m, 2),用于显示背景
        z_value: z高度值
        output_path: 输出图像路径(如果为None,则显示窗口）
        scale: 图像缩放因子(像素/单位）
        min_image_size: 最小图像尺寸(像素）,确保图像足够大
    """
    # 计算边界(包括路径和所有positions）
    all_points = np.vstack([path_xy, positions_xy])
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min

    # 创建图像(高度和宽度需要转换为整数像素）
    # 如果计算出的尺寸太小,自动调整scale以确保最小尺寸
    img_width = int(width * scale)
    img_height = int(height * scale)
    
    # 确保图像足够大
    if img_width < min_image_size or img_height < min_image_size:
        scale = max(min_image_size / width, min_image_size / height)
        img_width = int(width * scale)
        img_height = int(height * scale)
    
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # 将坐标转换为相对于图像原点的像素坐标
    # 先将世界坐标偏移,使得 x_min, y_min 对应到图像原点 (0, 0)
    def world_to_pixel(x, y):
        # 偏移坐标,使得最小值为0
        x_offset = x - x_min
        y_offset = y - y_min
        # 转换为像素坐标
        x_pixel = int(x_offset * scale)
        y_pixel = int(y_offset * scale)
        return x_pixel, y_pixel
    
    # 绘制所有free positions作为背景点(浅灰色小点）
    # 根据scale调整点的大小
    point_radius = max(1, int(scale / 50))
    for point in positions_xy:
        x_pixel, y_pixel = world_to_pixel(point[0], point[1])
        if 0 <= x_pixel < img_width and 0 <= y_pixel < img_height:
            cv2.circle(img, (x_pixel, y_pixel), point_radius, (200, 200, 200), -1)
    
    # 将路径点转换为像素坐标
    pixel_path = []
    for point in path_xy:
        x_pixel, y_pixel = world_to_pixel(point[0], point[1])
        pixel_path.append([x_pixel, y_pixel])
    
    pixel_path = np.array(pixel_path, dtype=np.int32)
    
    # 根据scale调整线条和点的大小
    line_thickness = max(2, int(scale / 20))
    start_end_radius = max(8, int(scale / 10))
    start_end_thickness = max(2, int(scale / 30))
    mid_point_radius = max(3, int(scale / 20))
    
    # 绘制路径线(使用渐变色,从蓝色到红色）
    for i in range(len(pixel_path) - 1):
        pt1 = tuple(pixel_path[i])
        pt2 = tuple(pixel_path[i + 1])
        color_ratio = i / (len(pixel_path) - 1) if len(pixel_path) > 1 else 0
        color = (
            int(255 * color_ratio),  # B
            128,  # G
            int(255 * (1 - color_ratio))  # R
        )
        cv2.line(img, pt1, pt2, color, line_thickness)
    
    # 绘制起点(绿色圆圈）
    start_pt = tuple(pixel_path[0])
    cv2.circle(img, start_pt, start_end_radius, (0, 255, 0), -1)
    cv2.circle(img, start_pt, start_end_radius, (0, 0, 0), start_end_thickness)
    
    # 绘制终点(红色圆圈）
    end_pt = tuple(pixel_path[-1])
    cv2.circle(img, end_pt, start_end_radius, (0, 0, 255), -1)
    cv2.circle(img, end_pt, start_end_radius, (0, 0, 0), start_end_thickness)
    
    # 绘制所有中间点(小圆点）
    for i in range(1, len(pixel_path) - 1):
        pt = tuple(pixel_path[i])
        cv2.circle(img, pt, mid_point_radius, (100, 100, 100), -1)
    
    # 根据图像尺寸调整字体大小和位置
    font_scale = max(0.6, min(2.0, scale / 30))
    font_thickness = max(1, int(font_scale * 2))
    text_spacing = int(30 * font_scale)
    
    # 添加文本标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = int(25 * font_scale)
    text_x = int(10 * font_scale)
    cv2.putText(img, f'Z Height: {z_value:.3f}', (text_x, y_offset), font, font_scale, (0, 0, 0), font_thickness)
    y_offset += text_spacing
    cv2.putText(img, f'Width: {width:.1f}', (text_x, y_offset), font, font_scale, (0, 0, 0), font_thickness)
    y_offset += text_spacing
    cv2.putText(img, f'Height: {height:.1f}', (text_x, y_offset), font, font_scale, (0, 0, 0), font_thickness)
    y_offset += text_spacing
    cv2.putText(img, f'Path Points: {len(path_xy)}', (text_x, y_offset), font, font_scale, (0, 0, 0), font_thickness)
    y_offset += text_spacing
    cv2.putText(img, f'Free Positions: {len(positions_xy)}', (text_x, y_offset), font, font_scale, (0, 0, 0), font_thickness)
    
    # 绘制坐标轴
    origin_x = int(50 * font_scale)
    origin_y = img_height - int(50 * font_scale)
    axis_length = int(100 * font_scale)
    axis_thickness = max(2, int(scale / 30))
    # X轴(红色）
    cv2.arrowedLine(img, (origin_x, origin_y), 
                    (origin_x + axis_length, origin_y), (0, 0, 255), axis_thickness)
    cv2.putText(img, 'X', (origin_x + axis_length + int(10 * font_scale), origin_y + int(5 * font_scale)), 
                font, font_scale, (0, 0, 255), font_thickness)
    # Y轴(绿色）
    cv2.arrowedLine(img, (origin_x, origin_y), 
                    (origin_x, origin_y - axis_length), (0, 255, 0), axis_thickness)
    cv2.putText(img, 'Y', (origin_x - int(20 * font_scale), origin_y - axis_length - int(10 * font_scale)), 
                font, font_scale, (0, 255, 0), font_thickness)
    
    cv2.imwrite(output_path, img)
    
    return img


def gen_path(
    free_position: str,
    same_z_height: bool = None,
    z_value: float = None,
    num_paths: int = 10,
    num_points: int = 30,
    max_angle_deviation: float = 45.0,
    z_tolerance: float = 0.0001,
    step_size: float = None,
    output_path: str = None,
    visualize: bool = True,
    vis_scale: float = 50.0,
    min_image_size: int = 2000,
    ):
    """
    从free positions中生成随机路径
    
    参数:
        free_position: free positions数组,形状为 (n, 4),包含 [x, y, z, semantic_label] 坐标
        same_z_height: 是否所有路径使用相同的z高度
        z_value: 指定的z高度值
        num_paths: 要生成的路径数量
        num_points: 每条路径的路径点数量
        max_angle_deviation: 最大角度偏差(度）,限制前进方向在前方左N度和右N度之间,默认45度
        z_tolerance: z值的容差(用于浮点数比较）
        step_size: 每步的最大步长(可选）
        output_path: 输出npy文件路径
        visualize: 是否可视化路径(生成图像）
        vis_scale: 可视化图像缩放因子(像素/单位）,默认50.0
        min_image_size: 最小图像尺寸(像素）,确保图像足够大,默认2000
    """
    # 1. 获取z值集合
    z_values = get_z_values_set(free_position)  # 形状为 (n,),包含所有唯一的z值,已经排序
    z_values = np.round(z_values, 3)  # 保留3位小数
    # 删除两个最大和两个最小Z值
    if len(z_values) > 4:
        z_values = np.array(z_values[2:-2])

    # 2. 为每条路径选择z高度
    if same_z_height or z_value is not None:
        # 所有路径使用相同的z高度
        if z_value is None:
            z_value = random.choice(z_values)   # 随机选择一个z高度
        else:
            z_idx = np.argmin(np.abs(z_values - z_value))  # 找到最接近的z高度
            z_value = z_values[z_idx]
        z_values_list = [z_value] * num_paths
    else:
        # 每条路径随机选择不同的z高度
        z_values_list = [random.choice(z_values) for _ in range(num_paths)] # 随机选择num_paths个不同的z高度
    
    # 3. 生成多条路径(每条路径在其对应的z高度上）
    paths_xy = []
    actual_z_values = []  # 保存成功生成的路径对应的z值
    all_positions_xy = []  # 用于可视化,收集所有z高度的positions
    for path_idx in range(num_paths):
        logger.info(f"生成路径 {path_idx + 1}/{num_paths}...")
        # 获取当前路径的z高度
        z_value = z_values_list[path_idx]
        
        # 筛选出该高度上的free positions
        positions_xy = filter_free_positions_at_z(free_position, z_value, z_tolerance)
        
        # 生成路径,形状为 (num_points, 2),包含 [x, y] 坐标
        path_xy = generate_random_path_from_positions(
            positions_xy,  # 形状为 (n, 2),包含 [x, y] 坐标
            num_points,  # 路径点的数量
            step_size,  # 每步的最大步长
            max_angle_deviation,  # 最大角度偏差
        )
        paths_xy.append(path_xy)  # 保存路径
        actual_z_values.append(z_value)  # 保存当前路径的z高度
        all_positions_xy.append(positions_xy)  # 保存该高度上的所有free positions
    
    # 4. 保存路径
    paths_xyz = save_paths_to_npy(paths_xy, actual_z_values, output_path)
    
    # 5. 可视化路径(如果启用）
    if visualize:
        vis_output_path = os.path.dirname(output_path)
        # 为每条路径生成单独的可视化图像
        for path_idx, (path_xy, z_val, positions_xy) in enumerate(zip(paths_xy, actual_z_values, all_positions_xy)):
            # 生成输出文件名
            vis_output_path_file = os.path.join(vis_output_path, f"{path_idx:04d}.png")
            visualize_path(path_xy, positions_xy, z_val, vis_output_path_file, vis_scale, min_image_size)

    return paths_xyz