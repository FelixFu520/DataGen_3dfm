#!/usr/bin/env python3
"""
从 USD 场景里，去掉所有 UsdGeom.Mesh 上已绑定的 OpenUSD(UsdPhysics) 与
Omniverse(PhysXSchema) 的物理/碰撞等 API 定义。

在已安装 OpenUSD 的环境运行（如 Isaac Sim 自带 Python: ./python.sh tools/remove_phy.py）。

用法:
  python tools/remove_phy.py
  python tools/remove_phy.py /path/to/scene.usd -o /path/to/out.usd
  python tools/remove_phy.py /path/to/scene.usd --in-place
"""
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

import argparse
import os
import sys
from typing import List, Type

# 导入Isaac Sim核心模块
import omni.replicator.core as rep
from isaacsim.asset.gen.omap.bindings import _omap  # 此文件不用, 但是别的文件要用, 这个要个 启动扩展 配合, 所以必须导入


try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError as e:
    print(
        "缺少 `pxr`（OpenUSD Python 绑定）。\n"
        "请使用含 OpenUSD 的 Python 环境执行，或先安装：\n"
        "  /home/fufa/projects2026/DataGen_3dfm/app/python.sh -m pip install usd-core\n"
        "然后重试该脚本。",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

# 默认要处理的资产（可改为自己的路径或命令行传参）
DEFAULT_USD = (
    "/home/fufa/projects2026/DataGen_3dfm/asset_extern/TaoBao/PostSovietKitchen/Kitchen.usd"
)


def _collect_usd_physics_api_classes() -> List[Type]:
    """从 UsdPhysics 模块中收集可 Remove 的 *API* schema 类。"""
    out: List[Type] = []
    # 与 Mesh/刚体/碰撞/材质/关节相关、常见于场景资产的类名
    names = [
        "CollisionAPI",
        "MeshCollisionAPI",
        "RigidBodyAPI",
        "MassAPI",
        "MaterialAPI",
        "ArticulationRootAPI",
        "DeformableBodyAPI",
        "FilteredPairsAPI",
        "DistanceJointAPI",
        "FixedJointAPI",
        "PrismaticJointAPI",
        "RevoluteJointAPI",
        "SphericalJointAPI",
        "LimitAPI",
        "DriveAPI",
    ]
    for n in names:
        c = getattr(UsdPhysics, n, None)
        if c is not None and isinstance(c, type):
            out.append(c)
    return out


def _collect_physx_api_classes() -> List[Type]:
    try:
        from pxr import PhysxSchema  # type: ignore
    except ImportError:
        return []
    out: List[Type] = []
    for n in dir(PhysxSchema):
        if not n.endswith("API") or n.startswith("_"):
            continue
        c = getattr(PhysxSchema, n, None)
        if c is not None and isinstance(c, type):
            out.append(c)
    return out


def _remove_one_schema(prim: Usd.Prim, api_cls: Type) -> bool:
    """从 prim 上移除单个 API（若已应用）。"""
    if not prim.HasAPI(api_cls):
        return False
    # USD 20+ 推荐
    if hasattr(prim, "RemoveAPI"):
        try:
            if prim.RemoveAPI(api_cls):
                return True
        except TypeError:
            # 部分多实例 API 需要 instance name
            if prim.RemoveAPI(api_cls, ""):
                return True
    unapply = getattr(api_cls, "Unapply", None)
    if callable(unapply):
        unapply(prim)
        return not prim.HasAPI(api_cls)
    return False


def strip_physics_from_prim(
    prim: Usd.Prim,
    all_api_classes: List[Type],
) -> int:
    """对单个 Prim 做彻底清理：移除 API + 物理命名空间属性。"""
    n = 0
    # 1) 先移除已知 Physics/PhysX API
    for api_cls in all_api_classes:
        if _remove_one_schema(prim, api_cls):
            n += 1

    # 2) 删除所有 physics/physx 相关属性和关系，避免残留
    # 常见命名空间：physics:* / physx* / physx:* / omni:physics:*
    remove_prefixes = (
        "physics:",
        "physx",
        "omni:physics",
    )
    for prop in prim.GetProperties():
        name = prop.GetName()
        lname = name.lower()
        if any(lname.startswith(pfx) for pfx in remove_prefixes):
            if prim.RemoveProperty(name):
                n += 1

    # 3) 兜底：移除任何包含 physics/physx 的 applied schema token
    #    例如 PhysicsCollisionAPI / PhysxSchemaPhysxCollisionAPI
    if hasattr(prim, "GetAppliedSchemas"):
        for schema_name in prim.GetAppliedSchemas():
            low = schema_name.lower()
            if "physics" in low or "physx" in low:
                if hasattr(prim, "RemoveAppliedSchema"):
                    if prim.RemoveAppliedSchema(schema_name):
                        n += 1
    return n


def run(usd_path: str, out_path: str | None, in_place: bool) -> None:
    if not os.path.isfile(usd_path):
        print(f"文件不存在: {usd_path}", file=sys.stderr)
        sys.exit(1)

    usd_physics = _collect_usd_physics_api_classes()
    physx = _collect_physx_api_classes()
    all_apis = usd_physics + physx

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"无法打开 USD: {usd_path}", file=sys.stderr)
        sys.exit(1)

    prim_count = 0
    touched = 0
    dirty_after_paths: List[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        prim_count += 1
        if strip_physics_from_prim(prim, all_apis) > 0:
            touched += 1
        # 二次检查：Prim 上仍存在 physics/physx 前缀属性则记录
        remain = []
        for prop in prim.GetProperties():
            n = prop.GetName().lower()
            if n.startswith("physics:") or n.startswith("physx") or n.startswith("omni:physics"):
                remain.append(prop.GetName())
        if remain:
            dirty_after_paths.append(f"{prim.GetPath()} -> {remain}")

    if in_place:
        out_path = usd_path
    if not out_path:
        base, _ = os.path.splitext(usd_path)
        out_path = f"{base}_nophy.usd"

    root = stage.GetRootLayer()
    if out_path == usd_path:
        root.Save()
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        root.Export(out_path)

    print(
        f"已处理 Prim 数: {prim_count}，曾含物理的 Prim: {touched}，"
        f"已写入: {out_path}"
    )
    if dirty_after_paths:
        print("警告：以下 Prim 仍检测到 physics/physx 属性（请发我这段输出我继续加规则）:")
        for row in dirty_after_paths:
            print(f"  {row}")


def main() -> None:
    p = argparse.ArgumentParser(description="从 USD 全场景 Prim 移除 UsdPhysics/PhysX 物理相关 API/属性")
    p.add_argument(
        "usd_path",
        nargs="?",
        default=DEFAULT_USD,
        help="输入 USD 路径",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出 USD 路径（与 --in-place 互斥，默认: <输入名>_nophy.usd）",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="直接覆盖原文件",
    )
    args = p.parse_args()
    if args.in_place and args.output:
        print("请只指定 --in-place 或 -o 之一", file=sys.stderr)
        sys.exit(1)
    run(args.usd_path, args.output, args.in_place)


if __name__ == "__main__":
    main()
# 关闭软件
simulation_app.close()