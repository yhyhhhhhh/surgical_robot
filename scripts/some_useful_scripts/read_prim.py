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
from pxr import Usd, UsdGeom, Gf

# ===================== 配置区（按需修改） =====================

# 要读取的 USD 文件
# USD_PATH = "/home/yhy/DVRK/ur3_scissor/object1.usd"
# USD_PATH = "/home/yhy/DVRK/ur3_scissor/ur3TipCam_pro1_1.usd"
# USD_PATH = "/media/yhy/PSSD/Isaac/4.0/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
# USD_PATH = "/home/yhy/DVRK/ur3_scissor/ur3TipCam_pro1_1_tweaked.usd"
# USD_PATH = "/home/yhy/DVRK/scenes/ur3_surgery_scene_flat.usd"
USD_PATH = "/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/pipe_stl.usd"
# 从哪个 prim 开始遍历：
#   - 例如 "/World/object"
#   - 如果设为 None，则优先使用 stage.defaultPrim，其次 "/World"，最后用 pseudoRoot
ROOT_PRIM_PATH = None

# 递归最大深度：
#   - None 表示不限制
#   - 0 表示只打印根 prim 本身
#   - 1 表示根 + 第一层孩子
MAX_DEPTH = None

# 是否计算并打印世界空间 bbox 尺寸
SHOW_BBOX = True

# 打印数组属性时，最多显示多少个元素（太长会截断）
MAX_ARRAY_ELEMS = 10

# ============================================================

import datetime

# ====== 配置：输出日志文件路径（按需修改）======
LOG_PATH = "/home/yhy/DVRK/ur3_scissor/pipe_stl_dump.txt"
# =================================================


# 打开日志文件（追加写入，或者改成 "w" 覆盖写）
_log_file = open(LOG_PATH, "w", encoding="utf-8")


def log_print(*args, **kwargs):
    """既打印到终端，又写入到日志文件。"""
    # 打印到终端
    print(*args, **kwargs)
    # 同样内容写到文件
    print(*args, **kwargs, file=_log_file)
    
# ============================================================

# 始终为 None 的属性（针对你这份 UR3 USD）
IGNORED_ATTRS_ALWAYS_NONE = {
    "outputs:displacement",
    "outputs:mdl:displacement",
    "outputs:mdl:surface",
    "outputs:mdl:volume",
    "outputs:surface",
    "outputs:volume",
    "outputs:out",
    "accelerations",
    "primvars:displayColor",
    "primvars:displayOpacity",
    "velocities",
}

# 始终为空列表 [] 的属性
IGNORED_ATTRS_ALWAYS_EMPTY = {
    "cornerIndices",
    "cornerSharpnesses",
    "creaseIndices",
    "creaseLengths",
    "creaseSharpnesses",
    "holeIndices",
}

# 始终没有 target 的关系
IGNORED_RELS_ALWAYS_EMPTY = {
    "proxyPrim",
    "physics:simulationOwner",
}


def should_skip_attr(attr: Usd.Attribute) -> bool:
    name = attr.GetName()

    # 1) 在「全程无用名单」里，直接跳过
    if name in IGNORED_ATTRS_ALWAYS_NONE or name in IGNORED_ATTRS_ALWAYS_EMPTY:
        return True

    # 2) 动态规则：值是 None 或空数组也跳过（更通用）
    try:
        v = attr.Get()
    except Exception:
        return False  # 取不到值就先别跳过，方便调试

    if v is None:
        return True

    # 空 list/tuple
    if isinstance(v, (list, tuple)) and len(v) == 0:
        return True

    return False


def should_skip_rel(rel: Usd.Relationship) -> bool:
    name = rel.GetName()

    # 1) 在「全程无用名单」里
    if name in IGNORED_RELS_ALWAYS_EMPTY:
        return True

    # 2) 动态规则：没有任何 target 的也跳过
    if not rel.GetTargets():
        return True

    return False
# ============================================================

def format_value(value, max_elems=10, max_str_len=200):
    """把 attribute 的值转成可读字符串，长数组会做截断。"""
    try:
        # 简单判断序列类型：list/tuple 等
        if isinstance(value, (list, tuple)):
            if len(value) > max_elems:
                head = list(value[:max_elems])
                tail = f"... (+{len(value) - max_elems} more)"
                return f"{head} {tail}"
            return str(list(value))
        # 其他类型直接转字符串，太长就截断
        s = str(value)
        if len(s) > max_str_len:
            return s[:max_str_len] + "...(truncated)"
        return s
    except Exception as e:
        return f"<error formatting value: {e}>"


def get_bound_material_paths(prim: Usd.Prim):
    """从 prim 的 relationships 里找所有材质绑定路径。"""
    paths = []
    for rel in prim.GetRelationships():
        name = rel.GetName()
        # 包含 material:binding 的都算（包括 material:binding:physics 等）
        if "material:binding" in name:
            for t in rel.GetTargets():
                paths.append(str(t))
    return paths


def print_prim_info(
    prim: Usd.Prim,
    bbox_cache=None,
    indent: int = 0,
    max_array_elems: int = 10,
    show_bbox: bool = True,
):
    """打印单个 prim 的详细信息。"""
    pad = "  " * indent
    log_print(
        f"{pad}Prim: {prim.GetPath()}  "
        f"type={prim.GetTypeName()}  "
        f"active={prim.IsActive()}  "
        f"instance={prim.IsInstance()}"
    )

    # 1) 世界空间 bbox 尺寸（如果需要且是 Boundable）
    if show_bbox and bbox_cache is not None and UsdGeom.Boundable(prim):
        try:
            bbox = bbox_cache.ComputeWorldBound(prim)
            aligned = bbox.ComputeAlignedBox()
            size = aligned.GetMax() - aligned.GetMin()
            log_print(
                f"{pad}  worldBBoxSize: "
                f"({size[0]:.6g}, {size[1]:.6g}, {size[2]:.6g})"
            )
        except Exception as e:
            log_print(f"{pad}  worldBBoxSize: <error: {e}>")

    # 2) Attributes
    for attr in prim.GetAttributes():
        if should_skip_attr(attr):
            continue  # 过滤掉没用的

        try:
            v = attr.Get()
        except Exception as e:
            v = f"<error getting value: {e}>"
        v_str = format_value(v, max_elems=max_array_elems)
        type_name = attr.GetTypeName()
        log_print(f"{pad}  attr {attr.GetName()} ({type_name}): {v_str}")

    # 3) Relationships
    for rel in prim.GetRelationships():
        if should_skip_rel(rel):
            continue  # 过滤掉没用的
        targets = [str(t) for t in rel.GetTargets()]
        log_print(f"{pad}  rel  {rel.GetName()} -> {targets}")

    # 4) 简要材质信息（绑定了哪些材质）
    mats = get_bound_material_paths(prim)
    if mats:
        log_print(f"{pad}  [materials] {mats}")

    log_print()  # 空行分隔一下


def traverse_subtree(
    root_prim: Usd.Prim,
    bbox_cache=None,
    indent: int = 0,
    max_depth: int | None = None,
    max_array_elems: int = 10,
    show_bbox: bool = True,
):
    """递归遍历并打印整个 prim 子树。"""
    print_prim_info(
        root_prim,
        bbox_cache=bbox_cache,
        indent=indent,
        max_array_elems=max_array_elems,
        show_bbox=show_bbox,
    )

    # 深度限制
    if max_depth is not None and indent >= max_depth:
        return

    for child in root_prim.GetChildren():
        traverse_subtree(
            child,
            bbox_cache=bbox_cache,
            indent=indent + 1,
            max_depth=max_depth,
            max_array_elems=max_array_elems,
            show_bbox=show_bbox,
        )


def main():
    # 打开 stage
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise RuntimeError(f"Failed to open stage: {USD_PATH}")

    log_print(f"Opened USD stage: {USD_PATH}")

    # Stage 基本信息
    log_print("\n=== Stage Info ===")
    log_print("DefaultPrim       :", stage.GetDefaultPrim())
    log_print("RootLayer         :", stage.GetRootLayer().identifier)
    log_print("UpAxis            :", UsdGeom.GetStageUpAxis(stage))
    meters_per_unit = stage.GetMetadata("metersPerUnit")
    if meters_per_unit is not None:
        log_print("metersPerUnit     :", meters_per_unit)
    log_print("TimeCodesPerSecond:", stage.GetTimeCodesPerSecond())
    log_print("==================\n")

    # 选择遍历的根 prim
    if ROOT_PRIM_PATH:
        root_prim = stage.GetPrimAtPath(ROOT_PRIM_PATH)
        if not root_prim:
            raise RuntimeError(f"Prim {ROOT_PRIM_PATH} not found in stage")
    else:
        # 先试 defaultPrim，其次 /World，再不行用 pseudoRoot
        if stage.GetDefaultPrim():
            root_prim = stage.GetDefaultPrim()
        else:
            world = stage.GetPrimAtPath("/World")
            if world:
                root_prim = world
            else:
                root_prim = stage.GetPseudoRoot()

    log_print(
        f"Root prim for traversal: {root_prim.GetPath()} "
        f"(type={root_prim.GetTypeName()})\n"
    )

    # BBox cache（可选）
    bbox_cache = None
    if SHOW_BBOX:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
            useExtentsHint=False,
        )

    # 遍历并打印整个子树
    traverse_subtree(
        root_prim,
        bbox_cache=bbox_cache,
        indent=0,
        max_depth=MAX_DEPTH,
        max_array_elems=MAX_ARRAY_ELEMS,
        show_bbox=SHOW_BBOX,
    )


if __name__ == "__main__":
    main()
