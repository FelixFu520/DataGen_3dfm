"""
3D 轨迹生成：
流程：
  1. 输入 free_xyz / occupied_xyz 体素点
  2. 3D 腐蚀 free_xyz（向内收缩 M 层）
  3. 用 occupied_xyz 做 3D flood-fill 求"室内/外包裹体"，将腐蚀后点过滤，
     不在室内的丢掉
  4. 在过滤后的 free 点中随机选起点，xy 方向保持 max_angle_deviation 平滑、
     z 方向用 max_dz_per_step 限制，按 step_size 做 3D 随机游走生成轨迹
  5. 将每条轨迹保存为 PLY（点云 + 线段）
"""

import os
import random
import numpy as np
from loguru import logger
from typing import List, Optional, Tuple

from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_erosion, binary_dilation, label


# ============================================================
# 栅格化工具：xyz 点 <-> 3D 体素网格
# ============================================================
class VoxelGrid3D:
    """将 xyz 世界坐标与 3D 体素索引互相转换的辅助类。"""

    def __init__(self, xyz_min: np.ndarray, resolution: float, shape: Tuple[int, int, int], padding: int = 0):
        self.xyz_min = xyz_min.astype(np.float64)
        self.resolution = float(resolution)
        self.shape = shape  # (nx, ny, nz)
        self.padding = padding

    @classmethod
    def from_points(cls, xyz: np.ndarray, resolution: float, padding: int = 2) -> "VoxelGrid3D":
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        nx = int(np.round((xyz_max[0] - xyz_min[0]) / resolution)) + 1 + 2 * padding
        ny = int(np.round((xyz_max[1] - xyz_min[1]) / resolution)) + 1 + 2 * padding
        nz = int(np.round((xyz_max[2] - xyz_min[2]) / resolution)) + 1 + 2 * padding
        return cls(xyz_min, resolution, (nx, ny, nz), padding)

    def world_to_index(self, xyz: np.ndarray) -> np.ndarray:
        """返回 (N, 3) int 索引 (ix, iy, iz)。"""
        idx = np.round((xyz - self.xyz_min) / self.resolution).astype(np.int64) + self.padding
        return idx

    def clip_index(self, idx: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.shape
        idx = idx.copy()
        idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
        return idx


# ============================================================
# 3D 腐蚀（对 free 体素在 xyz 三向均匀收缩）
# ============================================================
def erode_free_positions_3d(
    free_xyz: np.ndarray,
    occupied_xyz: Optional[np.ndarray],
    resolution: float,
    erode_iterations: int,
) -> np.ndarray:
    """
    将 free_xyz 栅格化为 3D 体素网格，对 free 体素做 3D 形态学腐蚀
    （3x3x3 结构元素迭代 erode_iterations 次），然后仅保留腐蚀后仍然存在的点。

    Args:
        free_xyz:         (N, 3) 或 (N, 4) free 点坐标，若带语义列只使用前3维
        occupied_xyz:     可选 (M, 3/4)，仅用来扩展体素网格范围，让腐蚀边界更稳定
        resolution:       与 occupancy 分辨率一致
        erode_iterations: 迭代次数，等于各方向收缩的体素层数
    Returns:
        (K, 3) 过滤后的点坐标
    """
    if erode_iterations <= 0 or len(free_xyz) == 0:
        return free_xyz[:, :3] if free_xyz.shape[1] > 3 else free_xyz

    free_xyz3 = free_xyz[:, :3]
    if occupied_xyz is not None and len(occupied_xyz) > 0:
        all_xyz = np.vstack([free_xyz3, occupied_xyz[:, :3]])
    else:
        all_xyz = free_xyz3

    grid = VoxelGrid3D.from_points(all_xyz, resolution, padding=erode_iterations + 2)
    nx, ny, nz = grid.shape

    volume = np.zeros((nx, ny, nz), dtype=bool)
    idx = grid.clip_index(grid.world_to_index(free_xyz3))
    volume[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # scipy 3D 腐蚀，structure=None 默认 3x3x3 十字形（6 邻域）；
    # 这里手工给一个 3x3x3 全 1 结构元素（26 邻域），收缩更均匀
    struct = np.ones((3, 3, 3), dtype=bool)
    eroded = binary_erosion(volume, structure=struct, iterations=erode_iterations)

    keep_mask = eroded[idx[:, 0], idx[:, 1], idx[:, 2]]
    filtered = free_xyz3[keep_mask]
    logger.info(
        f"3D 腐蚀过滤: {len(free_xyz3)} -> {len(filtered)} 点 "
        f"(移除 {len(free_xyz3) - len(filtered)} 个边缘点, iterations={erode_iterations})"
    )
    return filtered


# ============================================================
# 3D flood-fill 求室内（外包裹体）
# ============================================================
def filter_indoor_positions_3d(
    free_xyz: np.ndarray,
    occupied_xyz: np.ndarray,
    resolution: float,
    wall_dilate_iterations: int = 2,
) -> np.ndarray:
    """
    用 3D flood-fill 从 bbox 外部向内扩散，与外部连通的 free 体素视为屋外，
    不连通的（被墙壁围住）视为屋内。仅保留屋内的 free_xyz 点。

    Args:
        free_xyz:               (N, 3)
        occupied_xyz:           (M, 3)，用作墙壁
        resolution:             分辨率
        wall_dilate_iterations: 对墙壁做膨胀以填门窗缝隙
    Returns:
        (K, 3) 仅包含屋内的点
    """
    if len(free_xyz) == 0 or len(occupied_xyz) == 0:
        return free_xyz[:, :3] if free_xyz.shape[1] > 3 else free_xyz

    free_xyz3 = free_xyz[:, :3]
    occupied_xyz3 = occupied_xyz[:, :3]
    all_xyz = np.vstack([free_xyz3, occupied_xyz3])
    # padding 要够大，以保证 bbox 四周有一圈"屋外空气"供 flood-fill 作为种子
    padding = max(3, wall_dilate_iterations + 2)
    grid = VoxelGrid3D.from_points(all_xyz, resolution, padding=padding)
    nx, ny, nz = grid.shape

    wall = np.zeros((nx, ny, nz), dtype=bool)
    occ_idx = grid.clip_index(grid.world_to_index(occupied_xyz3))
    wall[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = True

    struct_full = np.ones((3, 3, 3), dtype=bool)
    if wall_dilate_iterations > 0:
        wall = binary_dilation(wall, structure=struct_full, iterations=wall_dilate_iterations)

    # "空气" = 非墙壁。用 26 连通做连通域标记，包含 bbox 边界的那一块连通域就是屋外
    air = ~wall
    # 6 连通更严格（更难"漏出去"），这里选 6 连通更接近 3D flood-fill 直觉
    struct_6 = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ], dtype=bool)
    labeled, num_components = label(air, structure=struct_6)

    # 收集所有与 bbox 边界接触的 label
    boundary_labels = set()
    for face in [
        labeled[0, :, :], labeled[-1, :, :],
        labeled[:, 0, :], labeled[:, -1, :],
        labeled[:, :, 0], labeled[:, :, -1],
    ]:
        boundary_labels.update(np.unique(face).tolist())
    boundary_labels.discard(0)

    # 屋外 mask = 属于任何边界连通域
    outdoor_mask = np.isin(labeled, list(boundary_labels))

    free_idx = grid.clip_index(grid.world_to_index(free_xyz3))
    # 室内 = 既不在"屋外连通域"，也不在"膨胀后的墙体"内
    # (墙体膨胀可能会吞噬紧贴外墙的 free 点, 这些点实际上在屋外, 需要一并剔除)
    in_outdoor = outdoor_mask[free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]]
    in_wall = wall[free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]]
    is_indoor = (~in_outdoor) & (~in_wall)
    filtered = free_xyz3[is_indoor]
    logger.info(
        f"3D 室内过滤: {len(free_xyz3)} -> {len(filtered)} 点 "
        f"(移除 {len(free_xyz3) - len(filtered)} 个屋外点, "
        f"wall_dilate={wall_dilate_iterations}, air_components={num_components})"
    )
    return filtered


# ============================================================
# 3D 随机游走生成轨迹
# ============================================================
def _pick_next_point_in_cone(
    current: np.ndarray,
    positions_xyz: np.ndarray,
    desired_angle: float,
    step_size_xy: float,
    dz_limit: float,
    cone_half_angle_rad: float,
    min_progress: float,
) -> Optional[int]:
    """在 current 的"前方锥形"区域里选下一个候选点。
    约束:
      - xy 距离 ∈ [min_progress, step_size_xy * 1.5]
      - z 距离 ≤ dz_limit
      - 相对于 desired_angle 的方位角偏差 ≤ cone_half_angle_rad
    返回落入锥内最靠近理想目标位置的点 idx, 没有则返回 None。
    """
    diff = positions_xyz - current
    dxy = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    dz = np.abs(diff[:, 2])

    in_range = (dxy >= min_progress) & (dxy <= step_size_xy * 1.5) & (dz <= dz_limit)
    if not np.any(in_range):
        return None

    # 方位角过滤
    cand_idx = np.where(in_range)[0]
    cand_diff = diff[cand_idx]
    cand_ang = np.arctan2(cand_diff[:, 1], cand_diff[:, 0])
    ang_err = np.abs(((cand_ang - desired_angle + np.pi) % (2 * np.pi)) - np.pi)
    cone_mask = ang_err <= cone_half_angle_rad
    if not np.any(cone_mask):
        return None

    cand_idx = cand_idx[cone_mask]
    # 在锥内, 偏好 xy 距离接近 step_size_xy 的点(进展大) + 方位角偏差小
    d_err = np.abs(dxy[cand_idx] - step_size_xy)
    score = d_err + step_size_xy * 0.5 * (ang_err[cone_mask] / max(cone_half_angle_rad, 1e-6))
    return int(cand_idx[int(np.argmin(score))])


def _pick_most_open_point(
    current: np.ndarray,
    positions_xyz: np.ndarray,
    radius: float,
    max_radius: float,
) -> int:
    """救援策略: 在 [radius, max_radius] 范围内, 挑选一个"周围 free 点数最多"的点,
    让后续能从这个开阔点继续游走。实现为在 radius-max_radius 环带中随机采样若干候选,
    选 density 最高者。"""
    diff = positions_xyz - current
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    band_mask = (dist >= radius) & (dist <= max_radius)
    if not np.any(band_mask):
        # 退而求其次: 在 max_radius 内任意一点
        band_mask = dist <= max_radius
        if not np.any(band_mask):
            return int(np.argmin(dist))

    cand = np.where(band_mask)[0]
    sample_size = min(64, len(cand))
    sampled = np.random.choice(cand, sample_size, replace=False) if len(cand) > sample_size else cand

    # density = 每个候选在小邻域内的 free 点数
    best_idx = int(sampled[0])
    best_density = -1
    nb_r = max(radius * 1.2, 0.3)
    for i in sampled:
        p = positions_xyz[i]
        d = np.sqrt(np.sum((positions_xyz - p) ** 2, axis=1))
        density = int(np.sum(d < nb_r))
        if density > best_density:
            best_density = density
            best_idx = int(i)
    return best_idx


def generate_random_path_3d(
    positions_xyz: np.ndarray,
    num_points: int,
    step_size_xy: float,
    step_size_z: float,
    max_angle_deviation: float,
    max_dz_per_step: float,
    stuck_window: int = 4,
    stuck_threshold_ratio: float = 0.5,
    rescue_max_factor: float = 8.0,
) -> np.ndarray:
    """
    在 3D free 点云里做随机游走, 带"反角落蜷缩"保护机制:
        - 起点: 从点集中随机选一个 (优先选周围开阔的点)
        - 每一步:
            1. 当前 xy 朝向 + [-max_angle_deviation, +max_angle_deviation] 随机扰动 → 期望朝向
            2. 在前方锥内(± max_angle_deviation + 一些余量)选一个满足 xy 进展 >= min_progress 的点
            3. 若锥内没有候选 -> 逐步放宽(加大锥角、降低 min_progress)
            4. 若连续 stuck_window 步累计 xy 位移 < stuck_threshold_ratio * step_xy * window
               -> 判定为"卡住", 随机大角度转向 (180°±90°) + 下一步锥角加大
            5. 若多轮放宽后仍失败, 或卡住超过 2 个 window, 触发救援:
               从 [2*step, rescue_max_factor*step] 环带里挑一个"周围最开阔"的点作为下一个点

    Args:
        positions_xyz:          (N, 3) 过滤后的可行 free 点
        num_points:             每条路径点数
        step_size_xy:           xy 每步目标步长
        step_size_z:            z 方向最大步长
        max_angle_deviation:    xy 朝向最大偏差(度), 也作为基础锥角
        max_dz_per_step:        每步 z 方向最大变化
        stuck_window:           用于卡住检测的滑动窗口长度
        stuck_threshold_ratio:  窗口内平均 xy 进展 < 该比例 * step 认为卡住
        rescue_max_factor:      救援环带最大半径 = step_size_xy * rescue_max_factor
    Returns:
        path_xyz: (num_points, 3)
    """
    if len(positions_xyz) == 0:
        raise ValueError("positions_xyz 为空, 无法生成路径")

    dz_limit = min(max_dz_per_step, step_size_z)
    base_cone = np.deg2rad(max_angle_deviation)
    min_progress_base = max(0.5 * step_size_xy, step_size_xy - 1e-6 * step_size_xy)
    # 放宽策略时, cone 和 min_progress 的逐步放宽系数
    cone_widen = [1.0, 2.0, 4.0, np.pi / max(base_cone, 1e-3)]  # 最后一档 = π (360° 任意)
    progress_shrink = [1.0, 0.6, 0.3, 0.15]

    # --- 起点: 优先选一个周围开阔的点, 而不是完全纯随机 ---
    rand_center = positions_xyz[random.randint(0, len(positions_xyz) - 1)]
    start_idx = _pick_most_open_point(
        rand_center, positions_xyz,
        radius=0.0, max_radius=step_size_xy * rescue_max_factor,
    )
    current = positions_xyz[start_idx].astype(np.float64)
    path = [current.copy()]

    current_angle = random.uniform(0, 2 * np.pi)
    recent_xy_moves: List[float] = []  # 最近 stuck_window 步的 xy 位移
    stuck_count = 0  # 连续 stuck window 计数

    for _ in range(num_points - 1):
        # 期望朝向 = 当前朝向 + 随机偏差
        angle_deviation = random.uniform(-base_cone, base_cone)
        desired_angle = (current_angle + angle_deviation) % (2 * np.pi)

        # 逐级放宽锥角和最小进展, 直到找到候选点
        nearest_idx = None
        for cw, ps in zip(cone_widen, progress_shrink):
            cone = min(np.pi, base_cone * cw)
            min_prog = min_progress_base * ps
            nearest_idx = _pick_next_point_in_cone(
                current, positions_xyz, desired_angle,
                step_size_xy=step_size_xy,
                dz_limit=dz_limit,
                cone_half_angle_rad=cone,
                min_progress=min_prog,
            )
            if nearest_idx is not None:
                break

        # --- 卡住检测 ---
        def window_stuck() -> bool:
            if len(recent_xy_moves) < stuck_window:
                return False
            avg_prog = float(np.mean(recent_xy_moves[-stuck_window:]))
            return avg_prog < stuck_threshold_ratio * step_size_xy

        need_rescue = (nearest_idx is None) or window_stuck()

        if need_rescue:
            stuck_count += 1
            # 先尝试"大角度转向": 在 180° 以内随机选一个新朝向再试一次
            new_angle = (current_angle + np.pi + random.uniform(-np.pi / 2, np.pi / 2)) % (2 * np.pi)
            retry_idx = _pick_next_point_in_cone(
                current, positions_xyz, new_angle,
                step_size_xy=step_size_xy,
                dz_limit=dz_limit,
                cone_half_angle_rad=np.pi,   # 任意方位
                min_progress=min_progress_base * 0.3,
            )
            if retry_idx is not None:
                nearest_idx = retry_idx
                logger.debug("轨迹卡住 -> 大角度转向成功")
            else:
                # 真跳不出来 -> 传送到一个"远且开阔"的点
                nearest_idx = _pick_most_open_point(
                    current, positions_xyz,
                    radius=step_size_xy * 2.0,
                    max_radius=step_size_xy * rescue_max_factor,
                )
                logger.debug("轨迹卡住 -> 救援传送到开阔点")
                recent_xy_moves.clear()
                stuck_count = 0
        else:
            stuck_count = 0

        nxt = positions_xyz[nearest_idx].astype(np.float64)
        dx = nxt[0] - current[0]
        dy = nxt[1] - current[1]
        recent_xy_moves.append(float(np.hypot(dx, dy)))
        if len(recent_xy_moves) > stuck_window * 2:
            recent_xy_moves = recent_xy_moves[-stuck_window * 2:]

        # 更新 xy 朝向; 若位移过小就不更新, 避免继续朝墙
        if np.hypot(dx, dy) > step_size_xy * 0.1:
            current_angle = np.arctan2(dy, dx)
            if current_angle < 0:
                current_angle += 2 * np.pi

        path.append(nxt.copy())
        current = nxt

    return np.asarray(path, dtype=np.float64)


# ============================================================
# PLY 可视化（点云 + 线段）
# ============================================================
def _sphere_points(center: np.ndarray, radius: float, num: int = 80) -> np.ndarray:
    """在半径为 radius 的球体内生成 num 个点(基于斐波那契球面 + 少量径向抖动)，
    用于把单个路径点"膨胀"成一个肉眼可见的小球。"""
    if num <= 0:
        return np.empty((0, 3), dtype=np.float64)
    indices = np.arange(num, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * indices / num)
    theta = np.pi * (1 + 5 ** 0.5) * indices  # 黄金角
    r = radius * (0.5 + 0.5 * np.cbrt(np.random.rand(num)))  # 随机径向抖动, 让球"实心"
    x = center[0] + r * np.sin(phi) * np.cos(theta)
    y = center[1] + r * np.sin(phi) * np.sin(theta)
    z = center[2] + r * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def _thick_segment_points(p0: np.ndarray, p1: np.ndarray, step: float, radius: float,
                          cross_samples: int = 8) -> np.ndarray:
    """把线段 p0->p1 离散成密集的"粗管"点，用于在 PLY 里画出肉眼可见的"粗线"。
    Args:
        step: 沿线方向采样步长(米)
        radius: 粗线半径(米)
        cross_samples: 每个位置在横截面上撒几个点
    """
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L <= 1e-9:
        return _sphere_points(p0, radius, num=cross_samples)
    n_along = max(2, int(np.ceil(L / step)) + 1)
    ts = np.linspace(0.0, 1.0, n_along)
    centers = p0[None, :] + ts[:, None] * v[None, :]  # (n_along, 3)

    # 构造两个与 v 正交的单位向量 e1, e2
    v_hat = v / L
    # 任取一个不平行于 v_hat 的向量
    if abs(v_hat[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(v_hat, tmp)
    e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(v_hat, e1)

    # 每个中心位置在横截面圆盘上撒 cross_samples 个点
    angles = np.linspace(0, 2 * np.pi, cross_samples, endpoint=False)
    rs = radius * (0.3 + 0.7 * np.sqrt(np.random.rand(cross_samples)))
    offsets = (rs[:, None] * np.cos(angles)[:, None] * e1[None, :] +
               rs[:, None] * np.sin(angles)[:, None] * e2[None, :])  # (cross, 3)

    # 广播叠加: (n_along, 1, 3) + (1, cross, 3) => (n_along, cross, 3)
    pts = centers[:, None, :] + offsets[None, :, :]
    return pts.reshape(-1, 3)


def _viridis_like_colormap(t: np.ndarray) -> np.ndarray:
    """简化的暖色渐变 (蓝 -> 青 -> 绿 -> 黄 -> 红), 用于标识路径点顺序。
    t: (N,) 取值 [0,1], 返回 (N,3) uint8"""
    t = np.clip(t, 0.0, 1.0)
    stops = np.array([
        [30, 60, 200],    # 起点: 深蓝
        [30, 180, 220],   # 青
        [60, 220, 80],    # 绿
        [255, 220, 40],   # 黄
        [230, 40, 40],    # 终点: 红
    ], dtype=np.float64)
    n = len(stops) - 1
    idx = t * n
    lo = np.clip(np.floor(idx).astype(int), 0, n - 1)
    hi = lo + 1
    frac = (idx - lo)[:, None]
    colors = stops[lo] * (1 - frac) + stops[hi] * frac
    return np.clip(colors, 0, 255).astype(np.uint8)


def save_path_ply(
    path_xyz: np.ndarray,
    free_xyz: Optional[np.ndarray],
    ply_path: str,
    background_color: Tuple[int, int, int] = (210, 210, 210),
    max_background_points: int = 50_000,
    line_radius: Optional[float] = None,
    line_step: Optional[float] = None,
    point_radius: Optional[float] = None,
    start_color: Tuple[int, int, int] = (0, 255, 0),
    end_color: Tuple[int, int, int] = (255, 0, 0),
):
    """
    保存单条 3D 路径到 PLY, 让轨迹在 MeshLab/CloudCompare 中肉眼可见:
      - 背景 free 点云: 浅灰, 下采样到 max_background_points
      - 路径"粗线": 将相邻点之间采样成密集"粗管", 沿路径从蓝->青->绿->黄->红渐变
      - 路径点"小球": 每个路径点膨胀成一个实心小球; 起点绿色大球, 终点红色大球
      - 同时保留 PLY 的 edge 元素, 兼容支持 edge 渲染的看图软件

    Args:
        path_xyz:              (N, 3)
        free_xyz:              背景点云, 或 None
        ply_path:              输出文件
        background_color:      背景点颜色
        max_background_points: 背景点上限, 默认 5 万(比之前更少, 突出轨迹)
        line_radius:           粗管半径(米), 默认根据场景尺度自适应
        line_step:             粗管沿线采样步长(米), 默认 = line_radius
        point_radius:          路径点小球半径(米), 默认 = 3 * line_radius
    """
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    path_xyz = np.asarray(path_xyz, dtype=np.float64)
    n_pts = len(path_xyz)

    # 自适应粗细: 基于路径整体 bbox 对角线
    if n_pts >= 2:
        bbox_diag = float(np.linalg.norm(path_xyz.max(axis=0) - path_xyz.min(axis=0)))
    else:
        bbox_diag = 1.0
    if line_radius is None:
        line_radius = max(0.02, bbox_diag * 0.006)  # 约 bbox 对角线的 0.6%
    if line_step is None:
        line_step = line_radius
    if point_radius is None:
        point_radius = line_radius * 3.0
    start_end_radius = point_radius * 1.5

    all_pts: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []

    # ---- 背景点云 ----
    if free_xyz is not None and len(free_xyz) > 0:
        bg = free_xyz[:, :3]
        if len(bg) > max_background_points:
            idx = np.random.choice(len(bg), max_background_points, replace=False)
            bg = bg[idx]
        bg_cols = np.tile(np.array(background_color, dtype=np.uint8), (len(bg), 1))
        all_pts.append(bg)
        all_cols.append(bg_cols)

    # ---- 路径"粗管"线段(按段着色, 颜色沿路径渐变) ----
    if n_pts >= 2:
        for i in range(n_pts - 1):
            t = i / max(1, n_pts - 2)  # 段的颜色取该段起点位置对应的渐变
            color = _viridis_like_colormap(np.array([t]))[0]
            seg_pts = _thick_segment_points(
                path_xyz[i], path_xyz[i + 1],
                step=line_step, radius=line_radius, cross_samples=8,
            )
            seg_cols = np.tile(color, (len(seg_pts), 1))
            all_pts.append(seg_pts)
            all_cols.append(seg_cols)

    # ---- 路径点"小球" ----
    t_pts = np.linspace(0.0, 1.0, n_pts) if n_pts > 1 else np.array([0.0])
    pt_colors = _viridis_like_colormap(t_pts)
    for i, p in enumerate(path_xyz):
        if i == 0:
            r = start_end_radius
            c = np.array(start_color, dtype=np.uint8)
        elif i == n_pts - 1:
            r = start_end_radius
            c = np.array(end_color, dtype=np.uint8)
        else:
            r = point_radius
            c = pt_colors[i]
        sph = _sphere_points(p, r, num=120)
        all_pts.append(sph)
        all_cols.append(np.tile(c, (len(sph), 1)))

    # ---- 合并全部 vertex ----
    if all_pts:
        pts = np.concatenate(all_pts, axis=0)
        cols = np.concatenate(all_cols, axis=0).astype(np.uint8)
    else:
        pts = np.empty((0, 3)); cols = np.empty((0, 3), dtype=np.uint8)

    # ---- 在 vertex 数组末尾再追加"纯路径点"专用索引, 供 edge 元素引用 ----
    edge_vertex_start = len(pts)
    if n_pts >= 2:
        extra = path_xyz.astype(np.float32)
        extra_cols = _viridis_like_colormap(np.linspace(0, 1, n_pts))
        pts = np.concatenate([pts, extra], axis=0)
        cols = np.concatenate([cols, extra_cols], axis=0)

    vertex_array = np.empty(len(pts), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex_array['x'] = pts[:, 0]
    vertex_array['y'] = pts[:, 1]
    vertex_array['z'] = pts[:, 2]
    vertex_array['red'] = cols[:, 0]
    vertex_array['green'] = cols[:, 1]
    vertex_array['blue'] = cols[:, 2]

    elements = [PlyElement.describe(vertex_array, 'vertex')]

    if n_pts >= 2:
        edges = np.empty(n_pts - 1, dtype=[
            ('vertex1', 'i4'), ('vertex2', 'i4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ])
        edge_cols = _viridis_like_colormap(np.linspace(0, 1, n_pts - 1))
        for i in range(n_pts - 1):
            edges[i] = (edge_vertex_start + i, edge_vertex_start + i + 1,
                        edge_cols[i, 0], edge_cols[i, 1], edge_cols[i, 2])
        elements.append(PlyElement.describe(edges, 'edge'))

    PlyData(elements, text=True).write(ply_path)


def save_filtered_points_ply(
    points_xyz: np.ndarray,
    ply_path: str,
    color: Tuple[int, int, int] = (100, 200, 100),
):
    """把过滤后的可行 free 点云保存为一个纯点云 PLY。"""
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    if len(points_xyz) == 0:
        logger.warning(f"{ply_path}: 点数为 0，跳过写入")
        return
    verts = np.empty(len(points_xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    verts['x'] = points_xyz[:, 0]
    verts['y'] = points_xyz[:, 1]
    verts['z'] = points_xyz[:, 2]
    verts['red'] = color[0]
    verts['green'] = color[1]
    verts['blue'] = color[2]
    PlyData([PlyElement.describe(verts, 'vertex')], text=True).write(ply_path)


# ============================================================
# 入口：生成多条 3D 路径
# ============================================================
def gen_path_3d(
    free_position: np.ndarray,
    occupied_position: np.ndarray,
    output_dir: str,
    num_paths: int = 10,
    num_points: int = 30,
    resolution: float = 0.1,
    erode_iterations: int = 3,
    wall_dilate_iterations: int = 2,
    step_size_xy: float = 0.3,
    step_size_z: float = 0.1,
    max_angle_deviation: float = 10.0,
    max_dz_per_step: float = 0.1,
    save_filtered_ply: bool = True,
) -> np.ndarray:
    """
    生成若干条 3D 轨迹，保存轨迹 npy 和每条轨迹的 PLY 可视化。

    流程：
      1) 3D 腐蚀 free_position
      2) 用 occupied_position 做 3D flood-fill，仅保留室内点
      3) 在过滤后的点云上做 3D 随机游走
      4) 输出 paths.npy、filtered_free_positions.ply（可选）、以及每条路径 {idx:04d}.ply

    Returns:
        paths_xyz: (num_paths, num_points, 3) ndarray
    """
    os.makedirs(output_dir, exist_ok=True)

    free_xyz = free_position[:, :3] if free_position.shape[1] > 3 else free_position
    occupied_xyz = occupied_position[:, :3] if occupied_position.shape[1] > 3 else occupied_position

    eroded_xyz = erode_free_positions_3d(
        free_xyz, occupied_xyz, resolution, erode_iterations
    )

    indoor_xyz = filter_indoor_positions_3d(
        eroded_xyz, occupied_xyz, resolution, wall_dilate_iterations
    )

    if len(indoor_xyz) == 0:
        raise RuntimeError(
            "过滤后点数为 0，无法生成路径。请检查 erode_iterations / "
            "wall_dilate_iterations / resolution 参数"
        )

    if save_filtered_ply:
        filtered_ply_path = os.path.join(output_dir, "filtered_free_positions.ply")
        save_filtered_points_ply(indoor_xyz, filtered_ply_path)
        logger.info(f"过滤后 free 点云已保存: {filtered_ply_path}, 点数: {len(indoor_xyz)}")

    paths_xyz: List[np.ndarray] = []
    for path_idx in range(num_paths):
        logger.info(f"生成 3D 路径 {path_idx + 1}/{num_paths}...")
        path_xyz = generate_random_path_3d(
            positions_xyz=indoor_xyz,
            num_points=num_points,
            step_size_xy=step_size_xy,
            step_size_z=step_size_z,
            max_angle_deviation=max_angle_deviation,
            max_dz_per_step=max_dz_per_step,
        )
        paths_xyz.append(path_xyz)

    paths_arr = np.stack(paths_xyz, axis=0)
    paths_npy_path = os.path.join(output_dir, "paths.npy")
    np.save(paths_npy_path, paths_arr)
    logger.info(f"路径已保存: {paths_npy_path}, 形状: {paths_arr.shape}")

    for path_idx, path_xyz in enumerate(paths_xyz):
        ply_path = os.path.join(output_dir, f"{path_idx:04d}.ply")
        save_path_ply(path_xyz, indoor_xyz, ply_path)
    logger.info(f"已保存 {len(paths_xyz)} 条路径的 PLY 可视化到 {output_dir}")

    return paths_arr
