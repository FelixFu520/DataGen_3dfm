from typing import List, Any
import os
import numpy as np
from PIL import Image
from pathlib import Path
from loguru import logger
# 尝试导入CuPy用于GPU加速
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy未安装，将使用CPU版本。安装CuPy以启用GPU加速: pip install cupy-cuda11x 或 cupy-cuda12x")

from pxr import (
    Gf,
    UsdGeom,
    Usd,
)
import omni.replicator.core as rep
from isaacsim.core.prims import XFormPrim
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.viewports import set_active_viewport_camera
from omni.isaac.core.utils.transformations import get_relative_transform

# OpenCV坐标系到Isaac Sim坐标系的转换矩阵
# OpenCV: X右, Y下, Z前 (相机朝向+Z)
# Isaac Sim: X前, Y右, Z上 (相机朝向-X)
# 转换关系: Isaac_X = -OpenCV_Z, Isaac_Y = -OpenCV_X, Isaac_Z = -OpenCV_Y
# 简化为: 翻转Y和Z轴
R_cv_to_isaac = np.eye(4)
R_cv_to_isaac[1, 1] = -1
R_cv_to_isaac[2, 2] = -1

CAMERA_CONFIGS = {
    "ZED_X.usdc": {
        "resolution": (1936, 1216),
        "left_camera_path": "base_link/ZED_X/CameraLeft",
        "right_camera_path": "base_link/ZED_X/CameraRight",
    },
    "zedx01.usd": {
        "resolution": (1936, 1216),
        "left_camera_path": "base_link/camera/CameraLeft",
        "right_camera_path": "base_link/camera/CameraRight",
    },
    "zedx02.usd": {
        "resolution": (1280, 800),
        "left_camera_path": "base_link/camera/CameraLeft",
        "right_camera_path": "base_link/camera/CameraRight",
    },
    "zedx03.usd": {
        "resolution": (1936, 1216),
        "left_bottom_camera_path": "base_link/camera/CameraLeftBottom",
        "right_bottom_camera_path": "base_link/camera/CameraRightBottom",
        "left_top_camera_path": "base_link/camera/CameraLeftTop",
        "right_top_camera_path": "base_link/camera/CameraRightTop",
    },
}

class Camera:
    """简化的相机类"""
    def __init__(self, camera_prim_path: str, resolution: tuple):
        self.prim_path = camera_prim_path
        self.resolution = resolution

        self.render_product = None
        self.LdrColor_annotator = None
        self.distance_to_camera_annotator = None
        self.distance_to_image_plane_annotator = None

    def _warmup_render_product(self, warmup_frames: int = 5):
        """创建 render_product 后先走几帧, 避免 Isaac Sim 5.1 已知 bug:
        LdrColor/distance annotator attach 时因 render_product 尚未初始化
        (size=0) 触发 'Unable to write from unknown dtype, kind=f, size=0'"""
        try:
            import omni.kit.app
            app = omni.kit.app.get_app()
            for _ in range(warmup_frames):
                app.update()
        except Exception as e:
            logger.warning(f"render_product warmup 失败, 忽略: {e}")

    def _attach_with_retry(self, annotator, render_product, name: str, max_retries: int = 5):
        """attach 失败时 step 几帧再重试, 规避 Isaac Sim 5.1 render_product 未就绪问题"""
        import omni.kit.app
        app = omni.kit.app.get_app()
        last_err = None
        for attempt in range(max_retries):
            try:
                annotator.attach(render_product)
                return
            except TypeError as e:
                last_err = e
                logger.warning(
                    f"{name} annotator attach 失败(第 {attempt + 1}/{max_retries} 次), "
                    f"render_product 可能未就绪, step 几帧后重试: {e}"
                )
                for _ in range(10):
                    app.update()
        raise RuntimeError(f"{name} annotator attach 重试 {max_retries} 次仍失败: {last_err}")

    def enable_rendering(self):
        """启用渲染"""
        if self.render_product is None:
            self.render_product = rep.create.render_product(self.prim_path, self.resolution, force_new=True)
            self._warmup_render_product()

    def enable_rgb(self):
        """启用RGB渲染"""
        if self.render_product is None:
            self.enable_rendering()

        if self.LdrColor_annotator is None:
            self.LdrColor_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
            self._attach_with_retry(self.LdrColor_annotator, self.render_product, "LdrColor")

    def enable_distance_to_camera(self):
        """启用深度渲染"""
        if self.render_product is None:
            self.enable_rendering()

        if self.distance_to_camera_annotator is None:
            self.distance_to_camera_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            self._attach_with_retry(self.distance_to_camera_annotator, self.render_product, "distance_to_camera")

    def enable_distance_to_image_plane(self):
        """启用深度渲染"""
        if self.render_product is None:
            self.enable_rendering()

        if self.distance_to_image_plane_annotator is None:
            self.distance_to_image_plane_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            self._attach_with_retry(self.distance_to_image_plane_annotator, self.render_product, "distance_to_image_plane")
    
    def enable_all(self):
        """启用所有渲染"""
        self.enable_rendering()
        self.enable_rgb()
        self.enable_distance_to_camera()
        self.enable_distance_to_image_plane()
    
    def get_rgb(self):
        """获取RGB图像"""
        if self.LdrColor_annotator is not None:
            return self.LdrColor_annotator.get_data()[:, :, :3]
        return None
        
    def get_distance_to_camera(self):
        """获取距离到相机图像"""
        if self.distance_to_camera_annotator is not None:
            return self.distance_to_camera_annotator.get_data()
        return None
    
    def get_distance_to_image_plane(self):
        """获取距离到图像平面图像"""
        if self.distance_to_image_plane_annotator is not None:
            return self.distance_to_image_plane_annotator.get_data()
        return None

    def get_camera_intrinsics(self, camera_prim):
        """获取相机内参"""
        camera_geom = UsdGeom.Camera(camera_prim)

        focal_length = camera_geom.GetFocalLengthAttr().Get()
        horizontal_aperture = camera_geom.GetHorizontalApertureAttr().Get()
        vertical_aperture = camera_geom.GetVerticalApertureAttr().Get()
        width, height = self.resolution

        fx = width * focal_length / horizontal_aperture
        fy = height * focal_length / vertical_aperture
        cx = width / 2.0
        cy = height / 2.0

        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_width(self):
        """获取宽度"""
        return self.resolution[0]
    
    def get_height(self):
        """获取高度"""
        return self.resolution[1]
    
    def __del__(self):
        """析构函数"""

        if self.LdrColor_annotator is not None:
            self.LdrColor_annotator.detach()
        if self.distance_to_camera_annotator is not None:
            self.distance_to_camera_annotator.detach()
        if self.distance_to_image_plane_annotator is not None:
            self.distance_to_image_plane_annotator.detach()
        if self.render_product is not None:
            self.render_product.destroy()

class CameraRig:
    """相机组类"""
    def __init__(self, camera_usd_url: str, world: World, stage: Usd.Stage):
        self.camera_usd_url = camera_usd_url
        self.camera_rig_name = Path(self.camera_usd_url).name

        self.world: World = world
        self.stage: Usd.Stage = stage

        self.camera_rig_prim_path = "/World/camera_rig" # 相机组prim路径
        self.camera_rig_prim = stage.DefinePrim(self.camera_rig_prim_path, "Xform") # 相机组prim对象
        
        self.cameras_name: List[str] = [] # 相机名称列表
        self.cameras_prim_path: List[str] = [] # 相机prim路径列表
        self.cameras_prim: List[Usd.Prim] = [] # 相机prim对象列表
        self.cameras: List[Camera] = [] # 相机对象列表

        logger.info("相机组名称: {}".format(self.camera_rig_name))

        self.init()
    
    def init(self):
        """初始化相机组"""
        if self.camera_rig_name == "ZED_X.usdc":
            # 添加相机USD
            camera_usd_path = self.camera_rig_prim_path + "/camera"
            add_reference_to_stage(usd_path=self.camera_usd_url, prim_path=camera_usd_path)

            # 创建XFormPrim对象来控制位姿
            xform_prim = XFormPrim(self.camera_rig_prim_path) # 相机组XFormPrim对象
            self.world.scene.add(xform_prim) # 将相机组XFormPrim对象添加到世界场景中

            # 相机prim路径
            left_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["left_camera_path"] # 左相机prim路径
            right_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["right_camera_path"] # 右相机prim路径
            self.cameras_prim_path.append(left_camera_prim_path) # 添加左相机prim路径到相机prim路径列表
            self.cameras_prim_path.append(right_camera_prim_path) # 添加右相机prim路径到相机prim路径列表

            # 获取相机prim对象
            left_camera_prim = self.stage.GetPrimAtPath(left_camera_prim_path) # 获取左相机prim对象
            right_camera_prim = self.stage.GetPrimAtPath(right_camera_prim_path) # 获取右相机prim对象
            self.cameras_prim.append(left_camera_prim) # 添加左相机prim对象到相机prim对象列表
            self.cameras_prim.append(right_camera_prim) # 添加右相机prim对象到相机prim对象列表

            # 创建相机对象
            left_camera = Camera(left_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建左相机对象
            right_camera = Camera(right_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建右相机对象
            self.cameras.append(left_camera) # 添加左相机对象到相机对象列表
            self.cameras.append(right_camera) # 添加右相机对象到相机对象列表

            # 相机名称
            self.cameras_name.append("left") # 添加左相机名称到相机名称列表
            self.cameras_name.append("right") # 添加右相机名称到相机名称列表

            # 设置相机视角
            set_active_viewport_camera(left_camera_prim_path)

            # 启用相机渲染
            left_camera.enable_all() # 启用左相机渲染
            right_camera.enable_all() # 启用右相机渲染
        elif self.camera_rig_name == "zedx01.usd":
           # 添加相机USD
            camera_usd_path = self.camera_rig_prim_path + "/camera"
            add_reference_to_stage(usd_path=self.camera_usd_url, prim_path=camera_usd_path)

            # 创建XFormPrim对象来控制位姿
            xform_prim = XFormPrim(self.camera_rig_prim_path) # 相机组XFormPrim对象
            self.world.scene.add(xform_prim) # 将相机组XFormPrim对象添加到世界场景中

            # 相机prim路径
            left_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["left_camera_path"] # 左相机prim路径
            right_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["right_camera_path"] # 右相机prim路径
            self.cameras_prim_path.append(left_camera_prim_path) # 添加左相机prim路径到相机prim路径列表
            self.cameras_prim_path.append(right_camera_prim_path) # 添加右相机prim路径到相机prim路径列表

            # 获取相机prim对象
            left_camera_prim = self.stage.GetPrimAtPath(left_camera_prim_path) # 获取左相机prim对象
            right_camera_prim = self.stage.GetPrimAtPath(right_camera_prim_path) # 获取右相机prim对象
            self.cameras_prim.append(left_camera_prim) # 添加左相机prim对象到相机prim对象列表
            self.cameras_prim.append(right_camera_prim) # 添加右相机prim对象到相机prim对象列表

            # 创建相机对象
            left_camera = Camera(left_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建左相机对象
            right_camera = Camera(right_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建右相机对象
            self.cameras.append(left_camera) # 添加左相机对象到相机对象列表
            self.cameras.append(right_camera) # 添加右相机对象到相机对象列表

            # 相机名称
            self.cameras_name.append("left") # 添加左相机名称到相机名称列表
            self.cameras_name.append("right") # 添加右相机名称到相机名称列表

            # 设置相机视角
            set_active_viewport_camera(left_camera_prim_path)

            # 启用相机渲染
            left_camera.enable_all() # 启用左相机渲染
            right_camera.enable_all() # 启用右相机渲染
        elif self.camera_rig_name == "zedx02.usd":
           # 添加相机USD
            camera_usd_path = self.camera_rig_prim_path + "/camera"
            add_reference_to_stage(usd_path=self.camera_usd_url, prim_path=camera_usd_path)

            # 创建XFormPrim对象来控制位姿
            xform_prim = XFormPrim(self.camera_rig_prim_path) # 相机组XFormPrim对象
            self.world.scene.add(xform_prim) # 将相机组XFormPrim对象添加到世界场景中

            # 相机prim路径
            left_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["left_camera_path"] # 左相机prim路径
            right_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["right_camera_path"] # 右相机prim路径
            self.cameras_prim_path.append(left_camera_prim_path) # 添加左相机prim路径到相机prim路径列表
            self.cameras_prim_path.append(right_camera_prim_path) # 添加右相机prim路径到相机prim路径列表

            # 获取相机prim对象
            left_camera_prim = self.stage.GetPrimAtPath(left_camera_prim_path) # 获取左相机prim对象
            right_camera_prim = self.stage.GetPrimAtPath(right_camera_prim_path) # 获取右相机prim对象
            self.cameras_prim.append(left_camera_prim) # 添加左相机prim对象到相机prim对象列表
            self.cameras_prim.append(right_camera_prim) # 添加右相机prim对象到相机prim对象列表

            # 创建相机对象
            left_camera = Camera(left_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建左相机对象
            right_camera = Camera(right_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建右相机对象
            self.cameras.append(left_camera) # 添加左相机对象到相机对象列表
            self.cameras.append(right_camera) # 添加右相机对象到相机对象列表

            # 相机名称
            self.cameras_name.append("left") # 添加左相机名称到相机名称列表
            self.cameras_name.append("right") # 添加右相机名称到相机名称列表

            # 设置相机视角
            set_active_viewport_camera(left_camera_prim_path)

            # 启用相机渲染
            left_camera.enable_all() # 启用左相机渲染
            right_camera.enable_all() # 启用右相机渲染
        elif self.camera_rig_name == "zedx03.usd":
           # 添加相机USD
            camera_usd_path = self.camera_rig_prim_path + "/camera"
            add_reference_to_stage(usd_path=self.camera_usd_url, prim_path=camera_usd_path)

            # 创建XFormPrim对象来控制位姿
            xform_prim = XFormPrim(self.camera_rig_prim_path) # 相机组XFormPrim对象
            self.world.scene.add(xform_prim) # 将相机组XFormPrim对象添加到世界场景中

            # 相机prim路径
            left_bottom_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["left_bottom_camera_path"] # 左相机prim路径
            right_bottom_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["right_bottom_camera_path"] # 右相机prim路径
            left_top_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["left_top_camera_path"] # 左相机prim路径
            right_top_camera_prim_path = camera_usd_path + "/" + CAMERA_CONFIGS[self.camera_rig_name]["right_top_camera_path"] # 右相机prim路径
            self.cameras_prim_path.append(left_bottom_camera_prim_path) # 添加左相机prim路径到相机prim路径列表
            self.cameras_prim_path.append(right_bottom_camera_prim_path) # 添加右相机prim路径到相机prim路径列表
            self.cameras_prim_path.append(left_top_camera_prim_path) # 添加左相机prim路径到相机prim路径列表
            self.cameras_prim_path.append(right_top_camera_prim_path) # 添加右相机prim路径到相机prim路径列表

            # 获取相机prim对象
            left_bottom_camera_prim = self.stage.GetPrimAtPath(left_bottom_camera_prim_path) # 获取左相机prim对象
            right_bottom_camera_prim = self.stage.GetPrimAtPath(right_bottom_camera_prim_path) # 获取右相机prim对象
            left_top_camera_prim = self.stage.GetPrimAtPath(left_top_camera_prim_path) # 获取左相机prim对象
            right_top_camera_prim = self.stage.GetPrimAtPath(right_top_camera_prim_path) # 获取右相机prim对象
            self.cameras_prim.append(left_bottom_camera_prim) # 添加左相机prim对象到相机prim对象列表
            self.cameras_prim.append(right_bottom_camera_prim) # 添加右相机prim对象到相机prim对象列表
            self.cameras_prim.append(left_top_camera_prim) # 添加左相机prim对象到相机prim对象列表
            self.cameras_prim.append(right_top_camera_prim) # 添加右相机prim对象到相机prim对象列表

            # 创建相机对象
            left_bottom_camera = Camera(left_bottom_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建左相机对象
            right_bottom_camera = Camera(right_bottom_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建右相机对象
            left_top_camera = Camera(left_top_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建左相机对象
            right_top_camera = Camera(right_top_camera_prim_path, CAMERA_CONFIGS[self.camera_rig_name]["resolution"]) # 创建右相机对象
            self.cameras.append(left_bottom_camera) # 添加左相机对象到相机对象列表
            self.cameras.append(right_bottom_camera) # 添加右相机对象到相机对象列表
            self.cameras.append(left_top_camera) # 添加左相机对象到相机对象列表
            self.cameras.append(right_top_camera) # 添加右相机对象到相机对象列表

            # 相机名称
            self.cameras_name.append("left_bottom") # 添加左相机名称到相机名称列表
            self.cameras_name.append("right_bottom") # 添加右相机名称到相机名称列表
            self.cameras_name.append("left_top") # 添加左相机名称到相机名称列表
            self.cameras_name.append("right_top") # 添加右相机名称到相机名称列表

            # 设置相机视角
            set_active_viewport_camera(left_bottom_camera_prim_path)

            # 启用相机渲染
            left_bottom_camera.enable_all() # 启用左相机渲染
            right_bottom_camera.enable_all() # 启用右相机渲染
            left_top_camera.enable_all() # 启用左相机渲染
            right_top_camera.enable_all() # 启用右相机渲染
        else:
            raise ValueError("不支持的相机 rig: {}".format(self.camera_rig_name))
    def set_camera_rig_pose(self, 
        x: float, y: float, z: float, 
        roll: float, pitch: float, yaw: float):
        """
        Args:   
            x: 相机位置x坐标
            y: 相机位置y坐标
            z: 相机位置z坐标
            roll: 相机roll角
            pitch: 相机pitch角
            yaw: 相机yaw角
        """
        camera_rig_prim = self.stage.GetPrimAtPath(self.camera_rig_prim_path)
        camera_xform = UsdGeom.Xformable(camera_rig_prim)
        camera_xform.ClearXformOpOrder()
        translate_op = camera_xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3f(x, y, z))
        rotate_op = camera_xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(roll, pitch, yaw))
    
    def get_cameras_name(self) -> List[str]:
        """获取相机名称"""
        return self.cameras_name
    
    def get_cameras_rgb(self) -> List[np.ndarray]:
        """获取RGB图像"""
        return [camera.get_rgb() for camera in self.cameras]
    
    def get_cameras_distance_to_image_plane(self) -> List[np.ndarray]:
        """获取距离到图像平面图像"""
        return [camera.get_distance_to_image_plane() for camera in self.cameras]
    
    def get_cameras_distance_to_camera(self) -> List[np.ndarray]:
        """获取距离到相机图像"""
        return [camera.get_distance_to_camera() for camera in self.cameras]
    
    def save_cameras_rgb(self, output_dir: str, path_idx: int = 0, point_idx: int = 0, 
        x: float = None, y: float = None, z: float = None,
        roll: float = None, pitch: float = None, yaw: float = None):
        """保存RGB图像"""
        for camera_name, camera_rgb in zip(self.cameras_name, self.get_cameras_rgb()):
            if x is not None and y is not None and z is not None and roll is not None and pitch is not None and yaw is not None:
                camera_rgb_path = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}_{x:.1f}_{y:.1f}_{z:.1f}_{roll:.1f}_{pitch:.1f}_{yaw:.1f}.jpg")
            else:
                camera_rgb_path = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}.jpg")

            camera_rgb_image = Image.fromarray(camera_rgb)

            os.makedirs(os.path.dirname(camera_rgb_path), exist_ok=True)
            camera_rgb_image.save(camera_rgb_path)

    def save_cameras_distance_to_image_plane(self, output_dir: str, path_idx: int = 0, point_idx: int = 0, 
        x: float = None, y: float = None, z: float = None,
        roll: float = None, pitch: float = None, yaw: float = None):
        """保存距离到图像平面图像"""
        for camera_name, depth_data in zip(self.cameras_name, self.get_cameras_distance_to_image_plane()):
            # 归一化可视化深度图
            is_not_inf_mask = depth_data != np.inf
            depth_normalized = (depth_data - depth_data[is_not_inf_mask].min()) / \
                (depth_data[is_not_inf_mask].max() - depth_data[is_not_inf_mask].min() + 1e-6)  # 深度图归一化

            # 保存可视化深度图
            if x is not None and y is not None and z is not None and roll is not None and pitch is not None and yaw is not None:
                depth_image_path = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}_{x:.1f}_{y:.1f}_{z:.1f}_{roll:.1f}_{pitch:.1f}_{yaw:.1f}.png")
            else:
                depth_image_path = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}.png")
            os.makedirs(os.path.dirname(depth_image_path), exist_ok=True)
            depth_image = Image.fromarray((np.where(is_not_inf_mask, depth_normalized * 255, 255)).astype(np.uint8))  # 深度图可视化
            depth_image.save(depth_image_path)

            # 保存深度数据
            if x is not None and y is not None and z is not None and roll is not None and pitch is not None and yaw is not None:
                depth_data_path_npy = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}_{x:.1f}_{y:.1f}_{z:.1f}_{roll:.1f}_{pitch:.1f}_{yaw:.1f}.npy")
            else:
                depth_data_path_npy = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}.npy")
            np.save(depth_data_path_npy, depth_data)

            # ===================== 保存压缩深度数据
            output_dir_compressed = os.path.join(os.path.dirname(output_dir), "depth_quantized")
            os.makedirs(output_dir_compressed, exist_ok=True)

            if x is not None and y is not None and z is not None and roll is not None and pitch is not None and yaw is not None:
                depth_image_path = os.path.join(output_dir_compressed, camera_name, f"{path_idx:04d}_{point_idx:04d}_{x:.1f}_{y:.1f}_{z:.1f}_{roll:.1f}_{pitch:.1f}_{yaw:.1f}.png")
            else:
                depth_image_path = os.path.join(output_dir_compressed, camera_name, f"{path_idx:04d}_{point_idx:04d}.png")
            os.makedirs(os.path.dirname(depth_image_path), exist_ok=True)
            depth_image.save(depth_image_path)

            depth_data_compressed = self._quantize_depth_data(depth_data)
            if x is not None and y is not None and z is not None and roll is not None and pitch is not None and yaw is not None:
                depth_data_compressed_path = os.path.join(output_dir_compressed, camera_name,f"{path_idx:04d}_{point_idx:04d}_{x:.1f}_{y:.1f}_{z:.1f}_{roll:.1f}_{pitch:.1f}_{yaw:.1f}.npz")
            else:
                depth_data_compressed_path = os.path.join(output_dir_compressed, camera_name,f"{path_idx:04d}_{point_idx:04d}.npz")

            os.makedirs(os.path.dirname(depth_data_compressed_path), exist_ok=True)
            self._save_quantized_file(depth_data_compressed_path, depth_data_compressed)
    
    def _save_quantized_file(self, out_path, codes: np.ndarray, *, decimals: int = 5, original_dtype: str = "float32") -> None:
        np.savez_compressed(
            out_path,
            data=codes,
            decimals=np.int32(decimals),
            current_type=np.array(np.int32),
            original_dtype=np.array(original_dtype),
        )
    
    def _quantize_depth_data(self, depth_data: np.ndarray, decimals: int = 5) -> np.ndarray:
        """
        将 float 深度量化为 int32: 值 = round(depth, decimals) * 10**decimals。
        在 float64 上算 round, 避免 float32 在 1e5 倍放大的边界误差。

        """
        scale = 10**decimals
        rounded = np.round(np.asarray(depth_data, dtype=np.float64), decimals)
        codes = np.rint(rounded * scale).astype(np.int32)
        return codes
    
    def save_cameras_distance_to_camera(self, output_dir: str, path_idx: int = 0, point_idx: int = 0, 
        x: float = None, y: float = None, z: float = None,
        roll: float = None, pitch: float = None, yaw: float = None):
        """保存距离到图像平面图像"""
        for camera_name, depth_data in zip(self.cameras_name, self.get_cameras_distance_to_camera()):
            # 归一化可视化深度图
            is_not_inf_mask = depth_data != np.inf
            depth_normalized = (depth_data - depth_data[is_not_inf_mask].min()) / \
                (depth_data[is_not_inf_mask].max() - depth_data[is_not_inf_mask].min() + 1e-6)  # 深度图归一化

            # 保存可视化深度图
            if x is not None and y is not None and z is not None and roll is not None and pitch is not None and yaw is not None:
                depth_image_path = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}_{x:.1f}_{y:.1f}_{z:.1f}_{roll:.1f}_{pitch:.1f}_{yaw:.1f}.png")
            else:
                depth_image_path = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}.png")
            os.makedirs(os.path.dirname(depth_image_path), exist_ok=True)
            depth_image = Image.fromarray((np.where(is_not_inf_mask, depth_normalized * 255, 255)).astype(np.uint8))  # 深度图可视化
            depth_image.save(depth_image_path)

            # 保存深度数据
            if roll is not None and pitch is not None and yaw is not None:
                depth_data_path_npy = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}_{roll:.1f}_{pitch:.1f}_{yaw:.1f}.npy")
            else:
                depth_data_path_npy = os.path.join(output_dir, camera_name, f"{path_idx:04d}_{point_idx:04d}.npy")
            np.save(depth_data_path_npy, depth_data)

    def get_cameras_intrinsics(self) -> List[np.ndarray]:
        """获取相机内参"""
        return [camera.get_camera_intrinsics(self.stage.GetPrimAtPath(camera.prim_path)) for camera in self.cameras]
    
    def get_camera_intrinsics(self, camera_idx: int) -> np.ndarray:
        """获取相机内参"""
        return self.get_cameras_intrinsics()[camera_idx]
    
    def get_camera_intrinsics_inv(self, camera_idx: int) -> np.ndarray:
        """获取相机内参逆矩阵"""
        return np.linalg.inv(self.get_camera_intrinsics(camera_idx))

    def get_transform_matrix(self, camera1_idx: int, camera2_idx: int) -> np.ndarray:
        """
        获取相机1到相机2的变换矩阵(4x4齐次矩阵,用于OpenCV坐标系)
        
        Returns:
            np.ndarray: 4x4变换矩阵,用法: P_camera2_homo = transform_matrix @ P_camera1_homo
        """
        # 使用Isaac Sim的get_relative_transform函数计算相机1到相机2的变换
        # get_relative_transform返回列主序(column-major)的4x4齐次变换矩阵
        transform_matrix_column_major = get_relative_transform(
            self.cameras_prim[camera1_idx],  # 源相机prim
            self.cameras_prim[camera2_idx]   # 目标相机prim
        )
        
        # 将R_cv_to_isaac变换整合到transform_matrix中
        # 变换顺序: 相机1坐标系(OpenCV) -> R_cv_to_isaac -> 相机1(USD) -> transform -> 相机2(USD) -> R_cv_to_isaac^T -> 相机2(OpenCV)
        # R_cv_to_isaac是对称矩阵(对角阵),所以 R_cv_to_isaac.T = R_cv_to_isaac^(-1) = R_cv_to_isaac
        transform_matrix = R_cv_to_isaac.T @ transform_matrix_column_major @ R_cv_to_isaac
        
        return transform_matrix

    def get_width(self) -> int:
        """获取宽度"""
        return self.cameras[0].get_width()
    
    def get_height(self) -> int:
        """获取高度"""
        return self.cameras[0].get_height()
    
    def get_camera_to_world_transform(self, camera_idx: int) -> np.ndarray:
        """
        获取相机坐标系到世界坐标系的变换矩阵(4x4齐次矩阵)
        
        注意: 这个变换矩阵适用于OpenCV坐标系下的点
        用法: P_world_homogeneous = transform_matrix @ P_camera_opencv_homogeneous
        """
        # 获取USD坐标系下的世界变换
        camera_xform = UsdGeom.Xformable(self.cameras_prim[camera_idx])
        usd_world_transform = camera_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # 转换为numpy数组 (USD使用行主序)
        usd_world_transform_np = np.array(usd_world_transform).reshape(4, 4).T  # 转置为列主序
        
        # 应用坐标系转换：OpenCV -> USD -> World
        # P_world = T_world_usd @ R_cv_to_isaac @ P_opencv
        opencv_to_world_transform = usd_world_transform_np @ R_cv_to_isaac

        return opencv_to_world_transform

    def cross_correspondence_numpy(self, camera1_idx: int, camera2_idx: int, threshold: float = 0.001) -> np.ndarray:
        """
        使用numpy向量化操作获取左右相机的交叉对应点, 左右目图像大小一致, left相当于相机1, right相当于相机2
        Args:
            camera1_idx: 相机1索引
            camera2_idx: 相机2索引
            threshold: 阈值
        Returns:
            np.ndarray: 交叉对应点坐标(u1, v1, u2, v2)列表
        """
        # 获取相机1和相机2的内参
        K_left = self.get_camera_intrinsics(camera1_idx)    # 左目内参矩阵
        K_right = self.get_camera_intrinsics(camera2_idx)    # 右目内参矩阵
        K_left_inv, K_right_inv = np.linalg.inv(K_left), np.linalg.inv(K_right)

        # 获取相机1和相机2的深度
        depth_left = self.get_cameras_distance_to_image_plane()[camera1_idx]    # 左目深度
        depth_right = self.get_cameras_distance_to_image_plane()[camera2_idx]    # 右目深度

        height = self.get_height()
        width = self.get_width()

        # 创建像素坐标网格 (height, width)
        v_grid, u_grid = np.mgrid[0:height, 0:width]
        
        # 展平为一维数组 (height*width,)
        u_left = u_grid.flatten()
        v_left = v_grid.flatten()
        depth_left_values = depth_left.flatten()

        # 过滤无效深度值
        valid_mask = (depth_left_values > 0) & np.isfinite(depth_left_values)
        u_left = u_left[valid_mask]
        v_left = v_left[valid_mask]
        depth_left_values = depth_left_values[valid_mask]

        # 构建齐次像素坐标 (N, 3)
        pixels_left = np.stack([u_left, v_left, np.ones_like(u_left)], axis=1)

        # 反投影: 左目像素 -> 左目相机坐标系3D点 (N, 3)
        P_left_camera = (K_left_inv @ pixels_left.T).T * depth_left_values[:, np.newaxis]

        # 构建齐次坐标 (N, 4)
        P_left_camera_homogeneous = np.concatenate([P_left_camera, np.ones((len(P_left_camera), 1))], axis=1)

        # 坐标系转换: 左目相机坐标系 -> 右目相机坐标系 (N, 4)
        transform_matrix = self.get_transform_matrix(camera1_idx, camera2_idx)
        left_points_in_right_camera_homogeneous = (transform_matrix @ P_left_camera_homogeneous.T).T

        # 转换回3D坐标 (N, 3)
        left_points_in_right_camera = left_points_in_right_camera_homogeneous[:, :3]

        # 过滤在右相机后方的点
        valid_mask = left_points_in_right_camera[:, 2] > 0
        u_left = u_left[valid_mask]
        v_left = v_left[valid_mask]
        P_left_camera_homogeneous = P_left_camera_homogeneous[valid_mask]
        left_points_in_right_camera = left_points_in_right_camera[valid_mask]

        # 重投影: 右目相机坐标系3D点 -> 右目像素 (N, 3)
        pixels_left_in_right = (K_right @ left_points_in_right_camera.T).T
        u_right = pixels_left_in_right[:, 0] / pixels_left_in_right[:, 2]
        v_right = pixels_left_in_right[:, 1] / pixels_left_in_right[:, 2]

        # 转换为整数像素坐标
        u_right_int = np.round(u_right).astype(int)
        v_right_int = np.round(v_right).astype(int)

        # 过滤超出右目图像范围的点
        valid_mask = (u_right_int >= 0) & (u_right_int < width) & (v_right_int >= 0) & (v_right_int < height)
        u_left = u_left[valid_mask]
        v_left = v_left[valid_mask]
        u_right_int = u_right_int[valid_mask]
        v_right_int = v_right_int[valid_mask]
        P_left_camera_homogeneous = P_left_camera_homogeneous[valid_mask]

        # 获取右目对应位置的深度值
        depth_right_values_actual = depth_right[v_right_int, u_right_int]

        # 过滤右目无效深度
        valid_mask = np.isfinite(depth_right_values_actual) & (depth_right_values_actual > 0)
        u_left = u_left[valid_mask]
        v_left = v_left[valid_mask]
        u_right_int = u_right_int[valid_mask]
        v_right_int = v_right_int[valid_mask]
        P_left_camera_homogeneous = P_left_camera_homogeneous[valid_mask]
        depth_right_values_actual = depth_right_values_actual[valid_mask]

        # 将左目3D点转换到世界坐标系 (N, 3)
        camera_to_world_left = self.get_camera_to_world_transform(camera1_idx)
        P_left_world = (camera_to_world_left @ P_left_camera_homogeneous.T).T[:, :3]

        # 从右目实际深度反投影得到右目相机坐标系中的3D点
        pixels_right_actual = np.stack([u_right_int, v_right_int, np.ones_like(u_right_int)], axis=1)
        P_right_camera_actual = (K_right_inv @ pixels_right_actual.T).T * depth_right_values_actual[:, np.newaxis]

        # 将右目3D点转换到世界坐标系 (N, 3)
        P_right_camera_actual_homogeneous = np.concatenate([P_right_camera_actual, np.ones((len(P_right_camera_actual), 1))], axis=1)
        camera_to_world_right = self.get_camera_to_world_transform(camera2_idx)
        P_right_world = (camera_to_world_right @ P_right_camera_actual_homogeneous.T).T[:, :3]

        # 计算世界坐标系中的欧式距离 (N,)
        euclidean_distances = np.linalg.norm(P_left_world - P_right_world, axis=1)

        # 过滤欧式距离过大的点
        valid_mask = euclidean_distances <= threshold
        u_left = u_left[valid_mask]
        v_left = v_left[valid_mask]
        u_right_int = u_right_int[valid_mask]
        v_right_int = v_right_int[valid_mask]

        # 构建最终的交叉对应点数组 (N, 4)
        cross_correspondences = np.stack([u_left, v_left, u_right_int, v_right_int], axis=1)

        return cross_correspondences

    def cross_correspondence_cpu(self, camera1_idx: int, camera2_idx: int, threshold: float = 0.001) -> np.ndarray:
        """
        获取左右相机的交叉对应点, 左右目图像大小一致, left相当于相机1, right相当于相机2
        Args:
            camera1_idx: 相机1索引
            camera2_idx: 相机2索引
            threshold: 阈值
        Returns:
            np.ndarray: 交叉对应点坐标(u1, v1, u2, v2)列表
        """
        # 获取相机1和相机2的内参
        K_left = self.get_camera_intrinsics(camera1_idx)    # 左目内参矩阵
        K_right = self.get_camera_intrinsics(camera2_idx)    # 右目内参矩阵
        K_left_inv, K_right_inv = np.linalg.inv(K_left), np.linalg.inv(K_right) # 左目、右目内参矩阵逆矩阵

        # 获取相机1和相机2的深度
        depth_left = self.get_cameras_distance_to_image_plane()[camera1_idx]    # 左目深度
        depth_right = self.get_cameras_distance_to_image_plane()[camera2_idx]    # 右目深度

        cross_correspondences = []

        for v_left in range(self.get_height()):
            for u_left in range(self.get_width()):
                # 获取左目深度值
                depth_left_value = depth_left[v_left, u_left]

                # 检查深度有效性
                if depth_left_value <= 0 or np.isinf(depth_left_value) or np.isnan(depth_left_value):
                    continue

                # 反投影: 左目像素 -> 左目相机坐标系3D点
                pixel_left = np.array([u_left, v_left, 1.0])
                P_left_camera = K_left_inv @ pixel_left * depth_left_value

                # 坐标系转换: 左目相机坐标系 -> 右目相机坐标系
                # 将3D点转换为齐次坐标 (1, 3) -> (1, 4)
                P_left_camera_homogeneous = np.append(P_left_camera, 1.0)
                # 应用变换矩阵
                left_points_in_right_camera_homogeneous = P_left_camera_homogeneous @ self.get_transform_matrix(camera1_idx, camera2_idx).T
                # 转换回3D坐标 (1, 4) -> (1, 3)
                left_points_in_right_camera = left_points_in_right_camera_homogeneous[:3]

                # 检查点是否在右相机前方
                if left_points_in_right_camera[2] <= 0:
                    continue

                # 重投影: 右目相机坐标系3D点 -> 右目像素
                pixel_left_in_right = K_right @ left_points_in_right_camera
                u_right = pixel_left_in_right[0] / pixel_left_in_right[2]
                v_right = pixel_left_in_right[1] / pixel_left_in_right[2]

                # 转换为整数像素坐标
                u_right_int = int(round(u_right))
                v_right_int = int(round(v_right))

                # 检查右目像素点是否在图像范围内
                if u_right_int < 0 or u_right_int >= self.get_width() or v_right_int < 0 or v_right_int >= self.get_height():
                    continue

                # 遮挡检测和有效性检查
                depth_right_value_actual = depth_right[v_right_int, u_right_int]

                # 检查右目深度有效性(过滤无限远和无效深度)
                if not np.isfinite(depth_right_value_actual) or depth_right_value_actual <= 0 or np.isinf(depth_right_value_actual):
                    continue  # 右目深度无效,跳过这个点

                # 将左目3D点转换到世界坐标系
                P_left_world = (self.get_camera_to_world_transform(camera1_idx) @ P_left_camera_homogeneous)[:3]

                # 从右目实际深度反投影得到右目相机坐标系中的3D点
                pixel_right_actual = np.array([u_right_int, v_right_int, 1.0])
                P_right_camera_actual = K_right_inv @ pixel_right_actual * depth_right_value_actual
                # 将右目3D点转换到世界坐标系
                P_right_camera_actual_homogeneous = np.append(P_right_camera_actual, 1.0)
                P_right_world = (self.get_camera_to_world_transform(camera2_idx) @ P_right_camera_actual_homogeneous)[:3]


                # 计算世界坐标系中的欧式距离
                # P_left_world 是从左目深度得到的世界坐标系3D点
                # P_right_world 是从右目深度得到的世界坐标系3D点
                # 如果两个点是同一个物理点,它们在世界坐标系中的欧式距离应该很小
                euclidean_distance = np.linalg.norm(P_left_world - P_right_world)

                # 通过世界坐标系中的欧式距离进行遮挡检测
                if euclidean_distance > threshold:
                    continue  # 欧式距离过大,左右目看到的不是同一个物理点,跳过

                cross_correspondences.append([u_left, v_left, u_right_int, v_right_int])


        return np.array(cross_correspondences)

    def cross_correspondence_cuda(self, camera1_idx: int, camera2_idx: int, threshold: float = 0.001) -> np.ndarray:
        """
        使用CUDA加速获取左右相机的交叉对应点, 左右目图像大小一致, left相当于相机1, right相当于相机2
        Args:
            camera1_idx: 相机1索引
            camera2_idx: 相机2索引
            threshold: 阈值
        Returns:
            np.ndarray: 交叉对应点坐标(u1, v1, u2, v2)列表
        """
        if not CUPY_AVAILABLE:
            logger.warning("CuPy不可用,回退到CPU版本")
            return self.cross_correspondence_cpu(camera1_idx, camera2_idx, threshold)
        
        # 获取相机参数(在CPU上)
        K_left = self.get_camera_intrinsics(camera1_idx)    # 左目内参矩阵
        K_right = self.get_camera_intrinsics(camera2_idx)    # 右目内参矩阵
        K_left_inv = np.linalg.inv(K_left)
        K_right_inv = np.linalg.inv(K_right)
        
        # 获取变换矩阵
        T_left_to_right = self.get_transform_matrix(camera1_idx, camera2_idx)
        T_left_to_world = self.get_camera_to_world_transform(camera1_idx)
        T_right_to_world = self.get_camera_to_world_transform(camera2_idx)
        
        # 获取深度图
        depth_left = self.get_cameras_distance_to_image_plane()[camera1_idx]
        depth_right = self.get_cameras_distance_to_image_plane()[camera2_idx]
        
        # 将数据传输到GPU (使用float64以保持精度一致)
        depth_left_gpu = cp.asarray(depth_left, dtype=cp.float64)
        depth_right_gpu = cp.asarray(depth_right, dtype=cp.float64)
        K_left_inv_gpu = cp.asarray(K_left_inv, dtype=cp.float64)
        K_right_inv_gpu = cp.asarray(K_right_inv, dtype=cp.float64)
        K_right_gpu = cp.asarray(K_right, dtype=cp.float64)
        T_left_to_right_gpu = cp.asarray(T_left_to_right, dtype=cp.float64)
        T_left_to_world_gpu = cp.asarray(T_left_to_world, dtype=cp.float64)
        T_right_to_world_gpu = cp.asarray(T_right_to_world, dtype=cp.float64)
        
        height, width = self.get_height(), self.get_width()
        
        # 创建像素网格 (使用int32以保持一致性)
        v_grid, u_grid = cp.meshgrid(cp.arange(height, dtype=cp.int32), 
                                      cp.arange(width, dtype=cp.int32), 
                                      indexing='ij')
        
        # 展平像素坐标
        u_flat = u_grid.ravel()
        v_flat = v_grid.ravel()
        depth_left_flat = depth_left_gpu.ravel()
        
        # 过滤有效深度点
        valid_mask = (depth_left_flat > 0) & cp.isfinite(depth_left_flat)
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_left_flat[valid_mask]
        
        if len(u_valid) == 0:
            return np.array([])
        
        # 反投影左目像素到3D点
        # 像素坐标 -> 归一化坐标 (转换为float64)
        pixel_coords = cp.stack([u_valid.astype(cp.float64), 
                                 v_valid.astype(cp.float64), 
                                 cp.ones(len(u_valid), dtype=cp.float64)], axis=1)  # (N, 3)
        P_left_camera = (K_left_inv_gpu @ pixel_coords.T).T * depth_valid[:, None]  # (N, 3)
        
        # 转换为齐次坐标
        P_left_camera_homo = cp.concatenate([P_left_camera, cp.ones((len(P_left_camera), 1), dtype=cp.float64)], axis=1)  # (N, 4)
        
        # 变换到右目相机坐标系
        P_right_camera_homo = (T_left_to_right_gpu @ P_left_camera_homo.T).T  # (N, 4)
        P_right_camera = P_right_camera_homo[:, :3]  # (N, 3)
        
        # 过滤在右相机前方的点
        valid_depth_mask = P_right_camera[:, 2] > 0
        P_right_camera = P_right_camera[valid_depth_mask]
        P_left_camera_homo = P_left_camera_homo[valid_depth_mask]
        u_valid = u_valid[valid_depth_mask]
        v_valid = v_valid[valid_depth_mask]
        
        if len(P_right_camera) == 0:
            return np.array([])
        
        # 投影到右目像素
        pixel_right = (K_right_gpu @ P_right_camera.T).T  # (N, 3)
        u_right = pixel_right[:, 0] / pixel_right[:, 2]
        v_right = pixel_right[:, 1] / pixel_right[:, 2]
        
        # 四舍五入到整数像素坐标
        u_right_int = cp.round(u_right).astype(cp.int32)
        v_right_int = cp.round(v_right).astype(cp.int32)
        
        # 过滤在图像范围内的点
        in_bounds_mask = (u_right_int >= 0) & (u_right_int < width) & \
                        (v_right_int >= 0) & (v_right_int < height)
        
        u_right_int = u_right_int[in_bounds_mask]
        v_right_int = v_right_int[in_bounds_mask]
        P_left_camera_homo = P_left_camera_homo[in_bounds_mask]
        u_valid = u_valid[in_bounds_mask]
        v_valid = v_valid[in_bounds_mask]
        
        if len(u_right_int) == 0:
            return np.array([])
        
        # 获取右目对应位置的深度值
        depth_right_actual = depth_right_gpu[v_right_int, u_right_int]
        
        # 过滤右目深度有效的点
        valid_right_depth_mask = (depth_right_actual > 0) & cp.isfinite(depth_right_actual)
        
        u_right_int = u_right_int[valid_right_depth_mask]
        v_right_int = v_right_int[valid_right_depth_mask]
        depth_right_actual = depth_right_actual[valid_right_depth_mask]
        P_left_camera_homo = P_left_camera_homo[valid_right_depth_mask]
        u_valid = u_valid[valid_right_depth_mask]
        v_valid = v_valid[valid_right_depth_mask]
        
        if len(u_right_int) == 0:
            return np.array([])
        
        # 计算左目点在世界坐标系中的位置
        P_left_world = (T_left_to_world_gpu @ P_left_camera_homo.T).T[:, :3]  # (N, 3)
        
        # 反投影右目像素到3D点
        pixel_right_actual = cp.stack([u_right_int.astype(cp.float64), 
                                       v_right_int.astype(cp.float64), 
                                       cp.ones(len(u_right_int), dtype=cp.float64)], axis=1)  # (N, 3)
        P_right_camera_actual = (K_right_inv_gpu @ pixel_right_actual.T).T * depth_right_actual[:, None]  # (N, 3)
        
        # 转换到世界坐标系
        P_right_camera_actual_homo = cp.concatenate([P_right_camera_actual, 
                                                     cp.ones((len(P_right_camera_actual), 1), dtype=cp.float64)], axis=1)
        P_right_world = (T_right_to_world_gpu @ P_right_camera_actual_homo.T).T[:, :3]  # (N, 3)
        
        # 计算欧式距离
        euclidean_distance = cp.linalg.norm(P_left_world - P_right_world, axis=1)
        
        # 过滤距离在阈值内的点
        valid_correspondence_mask = euclidean_distance <= threshold
        
        u_left_final = u_valid[valid_correspondence_mask]
        v_left_final = v_valid[valid_correspondence_mask]
        u_right_final = u_right_int[valid_correspondence_mask]
        v_right_final = v_right_int[valid_correspondence_mask]
        
        # 组合结果
        correspondences_gpu = cp.stack([u_left_final, v_left_final, u_right_final, v_right_final], axis=1)
        
        # 传回CPU
        correspondences = cp.asnumpy(correspondences_gpu).astype(np.int32)
        
        return correspondences