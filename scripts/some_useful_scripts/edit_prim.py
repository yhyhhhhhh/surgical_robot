import argparse

from omni.isaac.lab.app import AppLauncher

# -----------------------------------------------------------------------------
# 启动 Isaac/Kit
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Tune small object physics params in USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True   # 不需要 UI 的话就 True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------

from pxr import Usd, Sdf

# 原始 usd 路径
USD_PATH = "exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/object1.usd"
# 如果想另存一个版本，在这里填新路径；为 None 表示覆盖原文件
OUT_PATH = None


def get_or_create_attr(prim, name, type_name, default_value=None):
    """如果属性不存在则创建一个，再返回 attr。"""
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, type_name, custom=True)
        if default_value is not None:
            attr.Set(default_value)
    return attr


def main():
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise RuntimeError(f"Failed to open stage: {USD_PATH}")

    # ------------------------------------------------------------------
    # 1) 刚体 prim：/World/object
    # ------------------------------------------------------------------
    obj_prim = stage.GetPrimAtPath("/World/object")
    if not obj_prim:
        raise RuntimeError("Prim /World/object not found")

    print("=== /World/object ===")
    # 质量：调到 0.02 kg（20g），你可以改成 0.01~0.05 按效果微调
    mass_attr = get_or_create_attr(obj_prim, "physics:mass", Sdf.ValueTypeNames.Float)
    print("  physics:mass old:", mass_attr.Get())
    mass_attr.Set(0.001)
    print("  physics:mass new:", mass_attr.Get())

    # 线性/角阻尼：让物体更“黏”
    lin_damp_attr = get_or_create_attr(
        obj_prim, "physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float
    )
    ang_damp_attr = get_or_create_attr(
        obj_prim, "physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float
    )
    print("  linearDamping old:", lin_damp_attr.Get())
    lin_damp_attr.Set(3.0)
    print("  linearDamping new:", lin_damp_attr.Get())

    print("  angularDamping old:", ang_damp_attr.Get())
    ang_damp_attr.Set(3.0)
    print("  angularDamping new:", ang_damp_attr.Get())

    # 打开 CCD，防止小物体被穿透+弹飞
    ccd_attr = get_or_create_attr(
        obj_prim, "physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool
    )
    print("  enableCCD old:", ccd_attr.Get())
    ccd_attr.Set(True)
    print("  enableCCD new:", ccd_attr.Get())

    spec_ccd_attr = obj_prim.GetAttribute("physxRigidBody:enableSpeculativeCCD")
    if spec_ccd_attr:
        print("  enableSpeculativeCCD old:", spec_ccd_attr.Get())
        spec_ccd_attr.Set(True)
        print("  enableSpeculativeCCD new:", spec_ccd_attr.Get())
    else:
        print("  (no physxRigidBody:enableSpeculativeCCD attr; skip)")

    # ------------------------------------------------------------------
    # 2) Cube mesh prim：/World/object/Cube （碰撞相关参数）
    # ------------------------------------------------------------------
    cube_prim = stage.GetPrimAtPath("/World/object/Cube")
    if not cube_prim:
        raise RuntimeError("Prim /World/object/Cube not found")

    print("\n=== /World/object/Cube ===")
    contact_attr = get_or_create_attr(
        cube_prim, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
    )
    rest_attr = get_or_create_attr(
        cube_prim, "physxCollision:restOffset", Sdf.ValueTypeNames.Float
    )
    min_thickness_attr = get_or_create_attr(
        cube_prim, "physxConvexHullCollision:minThickness", Sdf.ValueTypeNames.Float
    )

    print("  contactOffset old:", contact_attr.Get())
    contact_attr.Set(1e-4)
    print("  contactOffset new:", contact_attr.Get())

    print("  restOffset old:", rest_attr.Get())
    rest_attr.Set(2e-5)
    print("  restOffset new:", rest_attr.Get())

    print("  minThickness old:", min_thickness_attr.Get())
    min_thickness_attr.Set(1e-4)
    print("  minThickness new:", min_thickness_attr.Get())

    # ------------------------------------------------------------------
    # 3) 物理材质：/World/PhysicsMaterial
    # ------------------------------------------------------------------
    phys_mat_prim = stage.GetPrimAtPath("/World/PhysicsMaterial")
    if not phys_mat_prim:
        print("\nWarning: /World/PhysicsMaterial not found")
    else:
        print("\n=== /World/PhysicsMaterial ===")

        def set_float_attr(name, value):
            attr = get_or_create_attr(phys_mat_prim, name, Sdf.ValueTypeNames.Float)
            print(f"  {name} old:", attr.Get())
            attr.Set(value)
            print(f"  {name} new:", attr.Get())

        # 高摩擦、零弹性
        set_float_attr("physics:staticFriction", 1.0)
        set_float_attr("physics:dynamicFriction", 0.8)
        set_float_attr("physics:restitution", 0.0)
        # 如果你想用材质来控制密度，也可以加：
        # set_float_attr("physics:density", 2000.0)

    # ------------------------------------------------------------------
    # 4) 保存
    # ------------------------------------------------------------------
    root_layer = stage.GetRootLayer()
    if OUT_PATH is None or OUT_PATH == USD_PATH:
        print(f"\nSaving back to original file: {USD_PATH}")
        root_layer.Save()
    else:
        print(f"\nExporting to new file: {OUT_PATH}")
        root_layer.Export(OUT_PATH)


if __name__ == "__main__":
    main()
