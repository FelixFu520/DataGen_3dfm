# 导入必要的库
import os
import sys
import time
import cv2
import argparse
import numpy as np
from loguru import logger
from typing import Tuple

# 配置日志
# 移除默认的日志处理器
logger.remove()  
# 配置日志格式
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
# 添加控制台输出
logger.add(
    sys.stdout,
    format="{message}",
    level="INFO",
)


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/home/fufa/d-isaacsim/_out_3dfm/scenes/interior_3dfm-camera-zedx")
parser.add_argument("--save_dir", type=str, default="./_out_3dfm/scenes/show")
parser.add_argument("--quantized", action="store_true", default=False, help="是否使用量化数据")
parser.add_argument("--show_num", type=int, default=10, help="要可视化的图片数量")
args = parser.parse_args()

def save_ply(filename, points, colors):
    """保存点云为PLY格式"""
    if len(points) == 0:
        logger.warning(f"点云为空，跳过保存: {filename}")
        return
    
    with open(filename, 'w') as f:
        # PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点云数据
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

def pixel2point_camera(depth: np.ndarray, image: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将像素坐标转换为相机坐标系下的3D点
    Args:
        depth: 深度图 (H, W)，单位：米
        image: RGB图像 (H, W, 3)
        K: 相机内参矩阵 (3, 3)
    Returns:
        points: 点云坐标 (N, 3) - 相机坐标系(OpenCV)
    """
    K_inv = np.linalg.inv(K)

    height, width = image.shape[:2]

    # 创建像素坐标网格 (height, width)
    v_grid, u_grid = np.mgrid[0:height, 0:width]

    # 展平为一维数组 (height*width,)
    u = u_grid.flatten()
    v = v_grid.flatten()
    depth_values = depth.flatten()

    # 过滤无效深度值
    valid_mask = (depth_values > 0) & np.isfinite(depth_values)
    u = u[valid_mask]
    v = v[valid_mask]
    depth_values = depth_values[valid_mask]

    # 构建齐次像素坐标 (N, 3)
    pixels = np.stack([u, v, np.ones_like(u)], axis=1)

    # 反投影: 像素 -> 相机坐标系3D点 (N, 3)
    points = (K_inv @ pixels.T).T * depth_values[:, np.newaxis]

    # 获取颜色（确保使用整数索引，并将BGR转换为RGB）
    v_int = v.astype(int)
    u_int = u.astype(int)
    colors = image[v_int, u_int, :3]

    return points, colors

def pixel2point_world(depth: np.ndarray, image: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将像素坐标转换为世界坐标系下的3D点
    Args:
        depth: 深度图 (H, W)，单位：米
        image: RGB图像 (H, W, 3)
        T_world_camera: 世界坐标系到相机坐标系的变换矩阵 (4, 4)
    Returns:
        points: 点云坐标 (N, 3) - 世界坐标系(OpenCV)

    """
    points, colors = pixel2point_camera(depth, image, K)
    points_world = (T_world_camera @ np.hstack([points, np.ones((len(points), 1))]).T).T[:, :3]
    return points_world, colors

def correspondence2point_camera(correspondence_data: np.ndarray, depth: np.ndarray, image: np.ndarray, K: np.ndarray, camera_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    将correspondence数据转换为相机坐标系下的3D点
    Args:
        correspondence_data: correspondence数据 (N, 4), 格式为 (u1, v1, u2, v2)
        depth: 深度图 (H, W)，单位：米
        image: RGB图像 (H, W, 3)
        K: 相机内参矩阵 (3, 3)
        camera_idx: 相机索引，0表示使用(u1, v1)，1表示使用(u2, v2)
    Returns:
        points: 点云坐标 (N, 3) - 相机坐标系(OpenCV)
        colors: 颜色 (N, 3)
    """
    if len(correspondence_data) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # correspondence_data格式: (u1, v1, u2, v2)
    # camera_idx=0: 使用(u1, v1)，camera_idx=1: 使用(u2, v2)
    if camera_idx == 0:
        u = correspondence_data[:, 0].astype(int)
        v = correspondence_data[:, 1].astype(int)
    else:
        u = correspondence_data[:, 2].astype(int)
        v = correspondence_data[:, 3].astype(int)
    
    # 过滤超出图像范围的点
    height, width = image.shape[:2]
    valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_mask]
    v = v[valid_mask]
    correspondence_data = correspondence_data[valid_mask]
    
    if len(u) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # 获取深度值
    depth_values = depth[v, u]
    
    # 过滤无效深度值
    valid_mask = (depth_values > 0) & np.isfinite(depth_values)
    u = u[valid_mask]
    v = v[valid_mask]
    depth_values = depth_values[valid_mask]
    correspondence_data = correspondence_data[valid_mask]
    
    if len(u) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # 构建齐次像素坐标 (N, 3)
    pixels = np.stack([u, v, np.ones_like(u)], axis=1)
    
    # 反投影: 像素 -> 相机坐标系3D点 (N, 3)
    K_inv = np.linalg.inv(K)
    points = (K_inv @ pixels.T).T * depth_values[:, np.newaxis]
    
    # 获取颜色
    colors = image[v, u, :3]
    
    return points, colors

def correspondence2point_world(correspondence_data: np.ndarray, depth: np.ndarray, image: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray, camera_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    将correspondence数据转换为世界坐标系下的3D点
    Args:
        correspondence_data: correspondence数据 (N, 4), 格式为 (u1, v1, u2, v2)
        depth: 深度图 (H, W)，单位：米
        image: RGB图像 (H, W, 3)
        K: 相机内参矩阵 (3, 3)
        T_world_camera: 世界坐标系到相机坐标系的变换矩阵 (4, 4)
        camera_idx: 相机索引，0表示使用(u1, v1)，1表示使用(u2, v2)
    Returns:
        points: 点云坐标 (N, 3) - 世界坐标系(OpenCV)
        colors: 颜色 (N, 3)
    """
    points, colors = correspondence2point_camera(correspondence_data, depth, image, K, camera_idx)
    if len(points) == 0:
        return points, colors
    points_world = (T_world_camera @ np.hstack([points, np.ones((len(points), 1))]).T).T[:, :3]
    return points_world, colors

if __name__ == "__main__":
    logger.info(f"args: {args}")
    os.makedirs(args.save_dir, exist_ok=True)

    # rbg dir
    rgb_dir = os.path.join(args.data_dir, "rgb")
    # depth dir
    if args.quantized:
        depth_dir = os.path.join(args.data_dir, "depth_quantized")
    else:
        depth_dir = os.path.join(args.data_dir, "depth")
    # crosscorrespondence dir
    if args.quantized:
        crosscorrespondence_dir = os.path.join(args.data_dir, "crosscorrespondence_quantized")
    else:
        crosscorrespondence_dir = os.path.join(args.data_dir, "crosscorrespondence")
    # common dir
    common_dir = os.path.join(args.data_dir, "common")

    # 相机名称 
    cameras_name = os.listdir(rgb_dir)
    assert len(cameras_name) > 0, f"相机名称列表为空: {cameras_name}"

    # 所有图片名称
    images_name = []
    for image_name in os.listdir(os.path.join(rgb_dir, cameras_name[0])):
        if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
            images_name.append(image_name)
    assert len(images_name) > 0, f"图像名称列表为空: {images_name}"
    images_name.sort()

    # 可视化所有图片, 并把图片投成点云可视化
    for image_idx, image_name in enumerate(images_name[:args.show_num]):  # 遍历所有图片
        logger.info(f"====> 可视化图片 {image_idx + 1}/{len(images_name)}: {image_name} <====")

        # 读取内外参
        common_path = os.path.join(common_dir, os.path.splitext(image_name)[0] + ".npy")
        common_dict = np.load(common_path, allow_pickle=True).item()

        # 读取correspondence数据
        if args.quantized:
            correspondence_path = os.path.join(crosscorrespondence_dir, os.path.splitext(image_name)[0] + ".npz")
            correspondence_dict = np.load(correspondence_path, allow_pickle=True)['data'].item()
        else:
            correspondence_path = os.path.join(crosscorrespondence_dir, os.path.splitext(image_name)[0] + ".npy")
            correspondence_dict = np.load(correspondence_path, allow_pickle=True).item()

        # 情况1: 相机的所有点图像
        for camera_name in cameras_name:    # 遍历所有相机
            # 读取图片
            image_path = os.path.join(rgb_dir, camera_name, image_name)
            image = cv2.imread(image_path)
            # OpenCV读取的是BGR格式，转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 确保图像是uint8类型，范围为[0, 255]
            image = image.astype(np.uint8)

            # 读取深度数据
            if args.quantized:
                depth_path = os.path.join(depth_dir, camera_name, os.path.splitext(image_name)[0] + ".npz")
                data_depth = np.load(depth_path, allow_pickle=True)
                depth = data_depth['data']
                decimals = data_depth['decimals']
                original_dtype = data_depth['original_dtype']

                scale = 10 ** decimals
                depth = (depth.astype(np.float64) / scale).astype(np.dtype(original_dtype.item()))
            else:
                depth_path = os.path.join(depth_dir, camera_name, os.path.splitext(image_name)[0] + ".npy")
                depth = np.load(depth_path)

            # 图片+深度+内参 ——> 相机坐标系下3D点
            points, colors = pixel2point_camera(depth, image, common_dict[camera_name]["intrinsics"])
            save_ply(os.path.join(args.save_dir, f"{os.path.splitext(image_name)[0]}_image_camera_{camera_name}.ply"), points, colors)

            # 图片+深度+内参+外参 ——> 世界坐标系下3D点
            points, colors = pixel2point_world(depth, image, common_dict[camera_name]["intrinsics"], common_dict[camera_name]["extrinsics_world"])
            save_ply(os.path.join(args.save_dir, f"{os.path.splitext(image_name)[0]}_image_world_{camera_name}.ply"), points, colors)

        # 情况2: 相机间的交叉点云
        for camera_pair in correspondence_dict.keys():
            camera1_name, camera2_name = camera_pair

            # 物理点在相机间的交叉对应点
            crosscorrespondence_data = correspondence_dict[camera_pair] # (N, 4), 左目像素坐标(u1, v1), 右目像素坐标(u2, v2)

            # 读取camera1相机图片
            image_path1 = os.path.join(rgb_dir, camera1_name, image_name)
            image1 = cv2.imread(image_path1)
            # OpenCV读取的是BGR格式，转换为RGB
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            # 确保图像是uint8类型，范围为[0, 255]
            image1 = image1.astype(np.uint8)

            # 读取camera2相机图片
            image_path2 = os.path.join(rgb_dir, camera2_name, image_name)
            image2 = cv2.imread(image_path2)
            # OpenCV读取的是BGR格式，转换为RGB
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            # 确保图像是uint8类型，范围为[0, 255]
            image2 = image2.astype(np.uint8)


            # 读取camera1相机深度数据
            if args.quantized:
                depth_path1 = os.path.join(depth_dir, camera1_name, os.path.splitext(image_name)[0] + ".npz")
                data_depth1 = np.load(depth_path1, allow_pickle=True)
                depth1 = data_depth1['data']
                decimals1 = data_depth1['decimals']
                original_dtype1 = data_depth1['original_dtype']
                scale1 = 10 ** decimals1
                depth1 = (depth1.astype(np.float64) / scale1).astype(np.dtype(original_dtype1.item()))
            else:
                depth_path1 = os.path.join(depth_dir, camera1_name, os.path.splitext(image_name)[0] + ".npy")
                depth1 = np.load(depth_path1)

            # 读取camera2相机深度数据
            if args.quantized:
                depth_path2 = os.path.join(depth_dir, camera2_name, os.path.splitext(image_name)[0] + ".npz")
                data_depth2 = np.load(depth_path2, allow_pickle=True)
                depth2 = data_depth2['data']
                decimals2 = data_depth2['decimals']
                original_dtype2 = data_depth2['original_dtype']
                scale2 = 10 ** decimals2
                depth2 = (depth2.astype(np.float64) / scale2).astype(np.dtype(original_dtype2.item()))
            else:
                depth_path2 = os.path.join(depth_dir, camera2_name, os.path.splitext(image_name)[0] + ".npy")
                depth2 = np.load(depth_path2)


            # 读取camera1相机和camera2相机的内参
            intrinsics1 = common_dict[camera1_name]["intrinsics"]
            intrinsics2 = common_dict[camera2_name]["intrinsics"]

            # 读取camera1相机和camera2相机的外参
            extrinsics1 = common_dict[camera1_name]["extrinsics_world"]
            extrinsics2 = common_dict[camera2_name]["extrinsics_world"]

            # 读取camera1相机到camera2相机的变换矩阵
            transform_matrix = common_dict[camera1_name]['extrinsics_camera'][camera2_name]

            save_ply_prefix = f"{os.path.splitext(image_name)[0]}_{camera_pair[0]}->{camera_pair[1]}"


            # ============ 交叉对应点+图片+深度+内参+外参 ——> 相机坐标系下3D点, 投影到camera1相机坐标系 ============
            # 使用camera1的像素坐标(u1, v1)和depth1
            corr_points_camera1, corr_colors_camera1 = correspondence2point_camera(
                crosscorrespondence_data, depth1, image1, intrinsics1, camera_idx=0
            )
            save_ply(os.path.join(args.save_dir, f"{save_ply_prefix}_camera_{camera1_name}.ply"), corr_points_camera1, corr_colors_camera1)


            # ============ 交叉对应点+图片+深度+内参+外参 ——> 相机坐标系下3D点, 投影到camera2相机坐标系 ============
            # 使用camera2的像素坐标(u2, v2)和depth2
            corr_points_camera2, corr_colors_camera2 = correspondence2point_camera(
                crosscorrespondence_data, depth2, image2, intrinsics2, camera_idx=1
            )
            save_ply(os.path.join(args.save_dir, f"{save_ply_prefix}_camera_{camera2_name}.ply"), corr_points_camera2, corr_colors_camera2)


            # ============ camera1相机坐标系下的点 投到 camera2相机坐标系下 ============
            corr_points_camera1_to_camera2 = (transform_matrix @ np.hstack([corr_points_camera1, np.ones((len(corr_points_camera1), 1))]).T).T[:, :3]
            save_ply(os.path.join(args.save_dir, f"{save_ply_prefix}_camera_{camera1_name}_in_{camera2_name}.ply"), corr_points_camera1_to_camera2, corr_colors_camera1)


            # ============ 交叉对应点+图片+深度+内参+外参 ——> 世界坐标系下3D点, 投影到camera1相机坐标系 ============
            # 使用camera1的像素坐标(u1, v1)和depth1
            corr_points_world1, corr_colors_world1 = correspondence2point_world(
                crosscorrespondence_data, depth1, image1, intrinsics1, extrinsics1, camera_idx=0
            )
            save_ply(os.path.join(args.save_dir, f"{save_ply_prefix}_world_{camera1_name}.ply"), corr_points_world1, corr_colors_world1)


            # ============ 交叉对应点+图片+深度+内参+外参 ——> 世界坐标系下3D点, 投影到camera2相机坐标系 ============
            # 使用camera2的像素坐标(u2, v2)和depth2
            corr_points_world2, corr_colors_world2 = correspondence2point_world(
                crosscorrespondence_data, depth2, image2, intrinsics2, extrinsics2, camera_idx=1
            )
            save_ply(os.path.join(args.save_dir, f"{save_ply_prefix}_world_{camera2_name}.ply"), corr_points_world2, corr_colors_world2)

            




