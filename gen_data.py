# 启动Isaac Sim应用
from isaacsim import SimulationApp
launch_config = {
    "headless": True,
    "renderer": "PathTracing",
    "rt_subframes": 128,
}
simulation_app = SimulationApp(launch_config=launch_config)

# 启动扩展
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.gen.omap")
simulation_app.update()

# 导入必要的库
import os
import sys
import random
import argparse
import numpy as np
from loguru import logger

# 导入Isaac Sim核心模块
import omni.replicator.core as rep
from isaacsim.asset.gen.omap.bindings import _omap  # 此文件不用, 但是别的文件要用, 这个要个 启动扩展 配合, 所以必须导入

# 自定义工具
from util.misc import load_usd_file
from util.occupancy import get_mesh_paths, get_semantic_occupancy, save_semantic_occupancy_ply
from util.random_path_3d import gen_path_3d
from util.camera import CameraRig
from util.misc_no_isaacsim import get_pair_combinations

# 解析命令行参数
parser = argparse.ArgumentParser()
# 环境参数
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--scene_usd_url", type=str, default="/home/fufa/d-isaacsim/asset_extern/interior_template_20251211/interior_template.usdc", help='场景USD文件路径')
parser.add_argument("--camera_usd_url", type=str, default="/home/fufa/projects2026/DataGen_3dfm/assets/ZED_X.usdc", help='相机USD文件路径')
parser.add_argument("--output_dir", type=str, default="/home/fufa/projects2026/DataGen_3dfm/workdir/3dfm/debug", help='输出目录')
# 生成occupancy所需参数
parser.add_argument("--occupancy_resolution", type=float, default=0.1, help='occupancy分辨率')
# 生成3D路径所需参数
parser.add_argument('--num_points', type=int, default=4, help='每条路径的路径点数量')
parser.add_argument('--num_paths', type=int, default=1, help='要生成的路径数量')
parser.add_argument('--max_angle_deviation', type=float, default=10.0, help='xy方向最大角度偏差(度),限制前进方向在前方左N度和右N度之间')
parser.add_argument('--erode_iterations', type=int, default=2, help='free positions 3D腐蚀迭代次数,越大过滤边缘越宽,设为0则不腐蚀')
parser.add_argument('--obstacle_dilate_iterations', type=int, default=2, help='occupied 障碍 3D 膨胀迭代次数(禁区),越大则 free 点离障碍越远(室内室外通用)')
parser.add_argument('--obstacle_envelope_iterations', type=int, default=0, help='occupied 障碍 3D 大膨胀迭代次数(活动包络体),仅保留落在包络内的 free 点,防止 free 点飞在半空/远离场景主体;设为0则关闭此过滤(室外场景建议开启)')
parser.add_argument('--step_size_xy', type=float, default=0.3, help='3D 路径 xy 方向每步最大步长(米)')
parser.add_argument('--step_size_z', type=float, default=0.1, help='3D 路径 z 方向每步最大步长(米)')
parser.add_argument('--max_dz_per_step', type=float, default=0.1, help='3D 路径相邻两点 z 方向最大变化(米)')

# 匹配参数
parser.add_argument("--occlusion_threshold", type=float, default=0.001, help="遮挡检测阈值(米), 空间两点欧式距离")
args = parser.parse_args()

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
    colorize=True
)
# 添加文件输出
logger.add(
    args.output_dir + "/gen_data.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    enqueue=True
)


if __name__ == "__main__":
    logger.info(f"args: {args}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ============ 步骤 1: 创建 World 和 Stage ============
    logger.info(f"步骤1: 加载场景, {args.scene_usd_url}")
    world, stage = load_usd_file(args.scene_usd_url)
    world.reset()   # 重置物理世界以确保场景完全加载
    for _ in range(60):
        simulation_app.update()
    
    # ============ 步骤 2: 生成occupancy ============
    logger.info("步骤 2: 生成occupancy")
    # logger.info("获取物理碰撞体(Mesh)的USD路径列表")
    mesh_paths = get_mesh_paths(stage)
    logger.info(f"共有mesh: {len(mesh_paths)}")

    logger.info("获取带语义的Occupancy")
    save_occupancy_dir = os.path.join(args.output_dir, "occupancy")
    os.makedirs(save_occupancy_dir, exist_ok=True)

    save_occupancy_occupied_npy_path = os.path.join(save_occupancy_dir, "occupied_positions.npy")
    save_occupancy_free_npy_path = os.path.join(save_occupancy_dir, "free_positions.npy")
    save_occupancy_occupied_ply_path = os.path.join(save_occupancy_dir, "occupied_positions.ply")
    save_occupancy_free_ply_path = os.path.join(save_occupancy_dir, "free_positions.ply")
    
    if not os.path.exists(save_occupancy_occupied_npy_path):
        semantic_occupancy = get_semantic_occupancy(stage, resolution=args.occupancy_resolution, mesh_paths=mesh_paths)
        occupied_data = semantic_occupancy[semantic_occupancy[:, 3] != 0]   # occupied positions
        free_data = semantic_occupancy[semantic_occupancy[:, 3] == 0]
        
        # save semantic_occupancy npy
        np.save(save_occupancy_occupied_npy_path, occupied_data)
        np.save(save_occupancy_free_npy_path, free_data)

        # save semantic_occupancy ply
        save_semantic_occupancy_ply(occupied_data, save_occupancy_occupied_ply_path)
        # save_semantic_occupancy_ply(free_data, save_occupancy_free_ply_path)
    else:
        occupied_data = np.load(save_occupancy_occupied_npy_path)
        free_data = np.load(save_occupancy_free_npy_path)

    logger.info(f"occupied_data: {occupied_data.shape}")
    logger.info(f"free_data: {free_data.shape}")

    # ============ 步骤 3: 生成 3D 路径 ============
    logger.info("步骤 3: 生成 3D 路径")
    output_path_dir = os.path.join(args.output_dir, "path")
    os.makedirs(output_path_dir, exist_ok=True)
    paths_xyz = gen_path_3d(
        free_position=free_data,
        occupied_position=occupied_data,
        output_dir=output_path_dir,
        num_paths=args.num_paths,
        num_points=args.num_points,
        resolution=args.occupancy_resolution,
        erode_iterations=args.erode_iterations,
        obstacle_dilate_iterations=args.obstacle_dilate_iterations,
        obstacle_envelope_iterations=args.obstacle_envelope_iterations,
        step_size_xy=args.step_size_xy,
        step_size_z=args.step_size_z,
        max_angle_deviation=args.max_angle_deviation,
        max_dz_per_step=args.max_dz_per_step,
        save_filtered_ply=True,
    )

    # ============ 步骤 4: 相机 ============
    logger.info("步骤 4: 添加相机")
    camera_rig = CameraRig(args.camera_usd_url, world, stage)
    # 等待渲染
    for i in range(60):
        world.step(render=True)
        rep.orchestrator.step()
        simulation_app.update()

    # ============ 步骤 5: 按照路径渲染, 并保存数据 ============
    save_rgb_dir = os.path.join(args.output_dir, "rgb") # 相机RGB图像
    save_rbg_discard_dir = os.path.join(args.output_dir, "rgb_discard") # 相机RGB图像被丢弃
    save_depth_dir = os.path.join(args.output_dir, "depth") # 相机距离到图像平面图像
    save_crosscorrespondence_dir = os.path.join(args.output_dir, "crosscorrespondence") # 相机交叉对应关系
    save_common_dir = os.path.join(args.output_dir, "common") # 相机外参, 内参等
    os.makedirs(save_rgb_dir, exist_ok=True)
    os.makedirs(save_rbg_discard_dir, exist_ok=True)
    os.makedirs(save_depth_dir, exist_ok=True)
    os.makedirs(save_crosscorrespondence_dir, exist_ok=True)
    os.makedirs(save_common_dir, exist_ok=True)
    for path_idx, path_xyz in enumerate(paths_xyz):
        for point_idx, point_xyz in enumerate(path_xyz):
            x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
            roll, pitch, yaw = 0, 0, 0

            logger.info(f"\n====> 生成 {path_idx:04d}_{point_idx:04d}(路径_路径点) 数据 <====")

            # 有时相机拍摄出来的图像是黑色的, 进入了角落, 所以需要尝试多次, 直到获取到正常的图像
            # 每次尝试在yaw旋转45度, 直到获取到正常的图像, 如果尝试了8次, 则认为相机进入了角落, 跳过这个点
            MAX_RETRY_ATTEMPTS = 8
            yaw_increment = 45
            retry_count = 0
            valid_image = False  # 此camera rig是否获取到全部正常的图像
            while retry_count < MAX_RETRY_ATTEMPTS:
                # 设置相机位置
                camera_rig.set_camera_rig_pose(x, y, z, roll, pitch, yaw)

                # 等待渲染
                for i in range(10):
                    world.step(render=True)
                    rep.orchestrator.step()
                    simulation_app.update()
                
                # 获取RGB图像
                cameras_rgb = camera_rig.get_cameras_rgb()
                # 保存RGB图像全部数据
                camera_rig.save_cameras_rgb(save_rbg_discard_dir, path_idx, point_idx, x, y, z, roll, pitch, yaw)

                # 用于判断相机是否在角落 是否保留
                valid_image_count = 0
                for camera_rgb in cameras_rgb:
                    # 排除整体偏暗的图片、排除色差小的图片
                    black_pixel_threshold = 10  # 整体偏暗的阈值
                    color_difference_threshold = 10  # 色差小的阈值
                    black_pixel_ratio = 0.5  # 整体偏暗的像素比例
                    black_pixel_count = np.sum(camera_rgb < black_pixel_threshold)
                    max_value = np.max(camera_rgb)  # 最大值
                    min_value = np.min(camera_rgb)  # 最小值
                    if black_pixel_count > black_pixel_ratio * camera_rgb.size or max_value - min_value < color_difference_threshold:
                        continue
                    else:
                        valid_image_count += 1
                
                # 根据camera rig中相机成像情况来判断是否旋转相机
                if valid_image_count < len(cameras_rgb):
                    logger.warning(f"采集 {path_idx:04d}-{point_idx:04d}-({x},{y},{z})-({roll},{pitch},{yaw}) 数据失败!!!!, 继续旋转相机, 格式:path-point-(xyz)-(roll,pitch,yaw)")
                    retry_count += 1
                    yaw += yaw_increment
                    continue
                else:
                    valid_image = True
                    break

            # 判断是否使用此点采集数据
            if not valid_image:
                logger.warning(f"采集 {path_idx:04d}-{point_idx:04d}-({x},{y},{z}) 数据失败!!!!, 重试 {retry_count} 次未成功, 跳过此点, 格式:path-point-(xyz)")
                continue
            else:
                logger.info(f"采集{path_idx:04d}-{point_idx:04d}-({x},{y},{z})) 数据成功, 重试 {retry_count} 次成功, 格式:path-point-(xyz)")
            
            # 获取相机组中所有相机名称
            cameras_name = camera_rig.get_cameras_name()

            # 获取相机组合
            cameras_combinations = get_pair_combinations(cameras_name)

            # 获取RGB数据, 并保存
            cameras_rgb = camera_rig.get_cameras_rgb()
            camera_rig.save_cameras_rgb(save_rgb_dir, path_idx, point_idx)

            # 获取距离到图像平面图像数据, 并保存
            cameras_distance_to_image_plane = camera_rig.get_cameras_distance_to_image_plane()
            camera_rig.save_cameras_distance_to_image_plane(save_depth_dir, path_idx, point_idx)

            # 获取correspondence数据, 并保存, correspondence 是world空间一点在左右相机图像平面上的对应点
            correspondence_data = {}
            for camera1_name, camera2_name in cameras_combinations:
                camera1_idx = cameras_name.index(camera1_name)
                camera2_idx = cameras_name.index(camera2_name)
                correspondence_data[(camera1_name, camera2_name)] = camera_rig.cross_correspondence_numpy(camera1_idx, camera2_idx)
            np.save(os.path.join(save_crosscorrespondence_dir, f"{path_idx:04d}_{point_idx:04d}.npy"), correspondence_data, allow_pickle=True)
            # ===================== 保存压缩深度数据
            correspondence_data_compressed_path = os.path.join(os.path.dirname(save_crosscorrespondence_dir), "crosscorrespondence_quantized")
            os.makedirs(correspondence_data_compressed_path, exist_ok=True)
            correspondence_data_compressed = {}
            for k, v in correspondence_data.items():
                correspondence_data_compressed[k] = correspondence_data[k].astype(np.int32)
            np.savez_compressed(
                os.path.join(correspondence_data_compressed_path, f"{path_idx:04d}_{point_idx:04d}.npz"),
                data=correspondence_data_compressed,
                current_type=np.array(np.int32),
                original_dtype=np.array(np.int64),
            )
            
            # 获取相机外参, 内参等数据, 并保存
            common_dict = {}
            for camera_name in cameras_name:
                camera_idx = cameras_name.index(camera_name)
                
                common_dict[camera_name] = {}

                # 内参
                common_dict[camera_name]["intrinsics"] = camera_rig.get_camera_intrinsics(camera_idx)

                # 外参(世界坐标系)
                common_dict[camera_name]["extrinsics_world"] = camera_rig.get_camera_to_world_transform(camera_idx)

                # 外参(相机坐标系)
                common_dict[camera_name]["extrinsics_camera"] = {}
                for camera_anthor in cameras_name:
                    if camera_anthor == camera_name:
                        continue
                    camera_anthor_idx = cameras_name.index(camera_anthor)
                    common_dict[camera_name]["extrinsics_camera"][camera_anthor] = camera_rig.get_transform_matrix(camera_idx, camera_anthor_idx)
            np.save(os.path.join(save_common_dir, f"{path_idx:04d}_{point_idx:04d}.npy"), common_dict, allow_pickle=True)
    

    logger.info(f"数据生成完成, 保存路径: {args.output_dir}")

    # 关闭软件
    simulation_app.close()