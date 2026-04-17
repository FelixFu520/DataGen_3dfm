from typing import List
import numpy as np
from loguru import logger
from plyfile import PlyData, PlyElement

import omni.physx
import omni.usd
from pxr import UsdGeom, Usd, UsdPhysics, Gf
from isaacsim.asset.gen.omap.bindings import _omap

from .misc_no_isaacsim import generate_high_contrast_colors


def get_mesh_paths(stage: Usd.Stage) -> List[str]:
    """
    获取物理碰撞体(Mesh)的USD路径列表, 仅返回包含物理碰撞体(Mesh)的USD路径
    注意: 如果Mesh没有物理碰撞体(Mesh), 则不会返回该USD路径

    Args:
        stage: Usd.Stage
    Returns:
        List[str]: 物理碰撞体(Mesh)的USD路径列表, 仅返回包含物理碰撞体(Mesh)的USD路径
    """
    mesh_paths = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            path = prim.GetPath()
            
            if UsdPhysics.CollisionAPI(prim) and UsdPhysics.MeshCollisionAPI(prim):
                mesh_paths.append(path)
            else:
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim)
    
    return mesh_paths

def get_semantic_occupancy(stage: Usd.Stage, resolution:float=0.05, mesh_paths: List[str] = None, margin_times:int=0) -> np.array:
    """
    获取带语义的Occupancy
    Args:
        stage: Usd.Stage
        resolution: 分辨率
        mesh_paths: 物理碰撞体(Mesh)的USD路径列表, 仅返回包含物理碰撞体(Mesh)的USD路径
        margin_times: 边缘裁剪/扩大倍数, 默认0, 即不裁剪边缘, 也不扩大
    Returns:
        np.array: 带语义的Occupancy
    """
    all_semantic_points = []
    
    physx = omni.physx.acquire_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # 计算所有 mesh 的联合包围盒，用于生成 free positions
    all_w_min = None
    all_w_max = None

    # 语义ID从1开始
    semantic_id = 1
    
    for path in mesh_paths:
        prim = stage.GetPrimAtPath(path)
        
        # 使用世界坐标 AABB 以确保完全覆盖旋转物体
        # 1. 获取世界坐标下的对齐包围盒 (AABB)
        # ComputeWorldBound 会自动计算应用了所有变换(旋转/缩放/位移)后的包围盒
        world_bbox = bbox_cache.ComputeWorldBound(prim)
        world_range = world_bbox.ComputeAlignedRange()
        
        w_min = world_range.GetMin()
        w_max = world_range.GetMax()
        w_center = world_range.GetMidpoint()
        
        # 2. 设置 Generator 参数
        # origin: 设置为包围盒中心，旋转保持为 0 (与世界坐标轴对齐)
        # 这样生成的体素网格是世界轴对齐的，能最稳定地包裹物体
        origin = (
            float(w_center[0]), float(w_center[1]), float(w_center[2]),
            0.0, 0.0, 0.0
        )
        
        # 3. 计算相对于 origin (中心点) 的局部边界
        # 增加 Margin 防止边缘裁剪
        margin = resolution * margin_times
        # lower 和 upper 是相对于 origin 的坐标
        lower = (
            float(w_min[0] - w_center[0] - margin), 
            float(w_min[1] - w_center[1] - margin), 
            float(w_min[2] - w_center[2] - margin)
        )
        upper = (
            float(w_max[0] - w_center[0] + margin), 
            float(w_max[1] - w_center[1] + margin), 
            float(w_max[2] - w_center[2] + margin)
        )

        # 4.调用生成器
        generator = _omap.Generator(physx, stage_id)
        # settings: voxel_size, occupied_thresh, free_thresh, unknown_thresh
        generator.update_settings(resolution, 1, 0, 255)
        # origin, lower, upper, 仅仅保留3位有效数字
        origin = (round(origin[0], 3), round(origin[1], 3), round(origin[2], 3), round(origin[3], 3), round(origin[4], 3), round(origin[5], 3))
        lower = (round(lower[0], 3), round(lower[1], 3), round(lower[2], 3))
        upper = (round(upper[0], 3), round(upper[1], 3), round(upper[2], 3))
        generator.set_transform(origin, lower, upper)
        
        generator.generate3d()
        pts_occupied = np.array(generator.get_occupied_positions()).astype(np.float32)
        pts_free = np.array(generator.get_free_positions()).astype(np.float32)
        
        if len(pts_occupied) > 10:
            # 拼接 ID: [x, y, z, semantic_id]
            ids = np.full((pts_occupied.shape[0], 1), semantic_id, dtype=np.float32)
            combined = np.hstack([pts_occupied, ids])
            all_semantic_points.append(combined)
            # 更新语义ID
            logger.info(f"成功标注: {path} -> ID: {semantic_id}, occupied点数: {len(pts_occupied)}, free点数: {len(pts_free)}")
            semantic_id += 1
        else:
            # logger.warning(f"警告: {path} 标注点数为 {len(pts_occupied)}, 跳过该 Mesh, 请检查该 Mesh 是否在物理层可见")
            continue

        # 5.更新联合包围盒
        if all_w_min is None:
            all_w_min = w_min
            all_w_max = w_max
        else:
            all_w_min = Gf.Vec3d(
                min(all_w_min[0], w_min[0]),
                min(all_w_min[1], w_min[1]),
                min(all_w_min[2], w_min[2])
            )
            all_w_max = Gf.Vec3d(
                max(all_w_max[0], w_max[0]),
                max(all_w_max[1], w_max[1]),
                max(all_w_max[2], w_max[2])
            )
        
    # 为整个场景生成 free positions（使用联合包围盒）
    logger.info(f"开始生成场景 free positions, 联合包围盒: {all_w_min} ~ {all_w_max}")
    if all_w_min is not None and len(mesh_paths) > 0:
        all_w_center = (all_w_min + all_w_max) / 2.0
        margin = resolution * margin_times  # 为 free space 增加更大的 margin
        
        origin_all = (
            float(all_w_center[0]), float(all_w_center[1]), float(all_w_center[2]),
            0.0, 0.0, 0.0
        )
        lower_all = (
            float(all_w_min[0] - all_w_center[0] - margin),
            float(all_w_min[1] - all_w_center[1] - margin),
            float(all_w_min[2] - all_w_center[2] - margin)
        )
        upper_all = (
            float(all_w_max[0] - all_w_center[0] + margin),
            float(all_w_max[1] - all_w_center[1] + margin),
            float(all_w_max[2] - all_w_center[2] + margin)
        )
        
        generator_all = _omap.Generator(physx, stage_id)
        generator_all.update_settings(resolution, 1, 0, 255)
        origin_all = (round(origin_all[0], 3), round(origin_all[1], 3), round(origin_all[2], 3), 
                     round(origin_all[3], 3), round(origin_all[4], 3), round(origin_all[5], 3))
        lower_all = (round(lower_all[0], 3), round(lower_all[1], 3), round(lower_all[2], 3))
        upper_all = (round(upper_all[0], 3), round(upper_all[1], 3), round(upper_all[2], 3))
        generator_all.set_transform(origin_all, lower_all, upper_all)
        
        generator_all.generate3d()
        pts_free_all = np.array(generator_all.get_free_positions()).astype(np.float32)
        
        if len(pts_free_all) > 0:
            # free positions 使用语义 ID 0
            ids_free = np.full((pts_free_all.shape[0], 1), 0, dtype=np.float32)
            combined_free = np.hstack([pts_free_all, ids_free])
            all_semantic_points.append(combined_free)
            logger.info(f"场景 free positions: {len(pts_free_all)} 个点, 语义 ID: 0")

    return np.vstack(all_semantic_points) if all_semantic_points else np.array([])

def save_semantic_occupancy_ply(semantic_occupancy: np.array, ply_path: str):
    """
    保存语义Occupancy为PLY文件
    Args:
        semantic_occupancy: 语义Occupancy
        path: PLY文件路径
    """
    vertices = []
    if int(np.max(semantic_occupancy[:, 3])) == 0:
        color_list = [(100, 150, 255)]  # free positions (ID=0) 使用浅蓝色，occupied positions 使用随机颜色
    else:
        color_list = generate_high_contrast_colors(int(np.max(semantic_occupancy[:, 3])))
    for p in semantic_occupancy:
        semantic_id = int(p[3])
        c = color_list[semantic_id % len(color_list)]
        vertices.append((p[0], p[1], p[2], c[0], c[1], c[2]))
        
    vertex_element = np.array(vertices, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    PlyData([PlyElement.describe(vertex_element, 'vertex')], text=True).write(ply_path)

def save_ply(filename, points, colors):
    """保存点云为PLY格式"""
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