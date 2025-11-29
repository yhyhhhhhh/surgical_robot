import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspect USD object: size, attributes, materials.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True  # set headless to False to see the UI
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


from pxr import Usd

# ============== 配置区：按需修改 =================
USD_PATH = "/home/yhy/DVRK/ur3_scissor/ur3TipCam_pro1_1.usd"
# 如果你想另存为新文件，而不是覆盖原文件，就把 OUT_PATH 改掉
OUT_PATH = "/home/yhy/DVRK/ur3_scissor/ur3TipCam_pro1_1_tweaked.usd"  # 比如 "/home/yhy/DVRK/ur3_scissor/ur3TipCam_pro1_1_tweaked.usd"

# 想改成的单位（推荐 1.0，表示 1 USD 单位 = 1 米）
NEW_METERS_PER_UNIT = 1.0
# =================================================


def set_attr(prim, name, value):
    """安全设置一个属性：打印旧值 -> 改新值。"""
    attr = prim.GetAttribute(name)
    if not attr:
        print(f"[WARN] {prim.GetPath()} 没有属性 {name}")
        return
    old = attr.Get()
    attr.Set(value)
    print(f"{prim.GetPath()}  {name}: {old} -> {attr.Get()}")


def main():
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise RuntimeError(f"无法打开 USD: {USD_PATH}")

    # 0) 修改 stage 的 metersPerUnit
    old_mpu = stage.GetMetadata("metersPerUnit")
    print("=== 修改 metersPerUnit ===")
    print("旧 metersPerUnit:", old_mpu)
    stage.SetMetadata("metersPerUnit", NEW_METERS_PER_UNIT)
    print("新 metersPerUnit:", stage.GetMetadata("metersPerUnit"))
    print()

    # 1) 修改 tip_joint 的驱动参数
    tip_joint_path = "/Root/ur3_robot/Extension_Link/tip_joint"
    tip_joint = stage.GetPrimAtPath(tip_joint_path)
    if not tip_joint:
        print(f"[WARN] 找不到 {tip_joint_path}")
    else:
        print("=== 调整 tip_joint 的刚度/阻尼/最大力 ===")
        # 比原来小很多，但仍然比较硬，你可以按需要再改
        set_attr(tip_joint, "drive:angular:physics:stiffness", 2000.0)
        set_attr(tip_joint, "drive:angular:physics:damping", 80.0)
        set_attr(tip_joint, "drive:angular:physics:maxForce", 100.0)
        print()

    # 2) 修 camera_link 碰撞 contactOffset
    cam_mesh_path = "/Root/ur3_robot/camera_link/mesh_"
    cam_mesh = stage.GetPrimAtPath(cam_mesh_path)
    if not cam_mesh:
        print(f"[WARN] 找不到 {cam_mesh_path}")
    else:
        print("=== 调整 camera_link/mesh_ 的 contactOffset ===")
        attr = cam_mesh.GetAttribute("physxCollision:contactOffset")
        if not attr:
            print(f"[WARN] {cam_mesh_path} 没有 physxCollision:contactOffset")
        else:
            old = attr.Get()
            print(f"{cam_mesh_path} physxCollision:contactOffset 旧值: {old}")
            attr.Set(5e-4)  # 0.0005 m
            print(f"{cam_mesh_path} physxCollision:contactOffset 新值: {attr.Get()}")
        print()

    # 3) （可选）打开自碰撞
    root_joint_path = "/Root/ur3_robot/root_joint"
    root_joint = stage.GetPrimAtPath(root_joint_path)
    if not root_joint:
        print(f"[WARN] 找不到 {root_joint_path}")
    else:
        print("=== （可选）打开 UR3 自碰撞 ===")
        esc_attr = root_joint.GetAttribute("physxArticulation:enabledSelfCollisions")
        if esc_attr:
            old = esc_attr.Get()
            # 如果暂时不想开，可以注释掉下一行
            # esc_attr.Set(True)
            print(f"{root_joint_path} enabledSelfCollisions: {old} -> {esc_attr.Get()}")
        else:
            print(f"[WARN] {root_joint_path} 没有 enabledSelfCollisions 属性")
        print()

    # 4) 保存
    root_layer = stage.GetRootLayer()
    if OUT_PATH is None or OUT_PATH == USD_PATH:
        print(f"保存到原文件: {USD_PATH}")
        root_layer.Save()
    else:
        print(f"导出到新文件: {OUT_PATH}")
        root_layer.Export(OUT_PATH)


if __name__ == "__main__":
    main()
