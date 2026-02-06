
import re
import math

# ========= 配置：修改成你自己的 txt 路径 =========
INPUT_PATH = "/home/yhy/DVRK/ur3_scissor/ur3_surgery_scene_flat_dump.txt"
OUTPUT_PATH = "/home/yhy/DVRK/scenes/ur3_surgery_scene_flat_dump_physics_only.txt"
# ===============================================


def is_interesting_attr_name(name: str) -> bool:
    """只保留和物理 / 位姿 / 尺度相关的属性名."""
    prefix_keep = (
        "physics:",
        "physx",
        "drive:",
        "state:",
        "xformOp:translate",
        "xformOp:scale",
        "xformOp:orient",
    )
    if any(name.startswith(p) for p in prefix_keep):
        return True
    # 这些辅助信息也可以保留一点，方便看
    if name in ("extent", "purpose", "visibility"):
        return True
    return False


def warning_for_attr(name: str, value_str: str) -> str:
    """根据文本里的值做一点简单体检，返回中文提示（没有问题就返回空字符串）."""
    note = ""

    # 只看字符串，不做太复杂分析
    if "inf" in value_str or "nan" in value_str.lower():
        note += "值包含 inf/NaN，通常说明还没设置好；建议检查。"

    # 质量
    if name == "physics:mass":
        try:
            v = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_str)[0])
        except Exception:
            v = None
        if v is not None:
            if v <= 0:
                note += " 质量 <= 0，可能不合理。"
            elif v < 1e-6:
                note += " 质量非常小，可能导致数值不稳定。"

    # 碰撞壳厚度 contactOffset
    if name == "physxCollision:contactOffset":
        try:
            v = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_str)[0])
        except Exception:
            v = None
        if v is not None:
            if v <= 0:
                note += " contactOffset 非正，建议改成略大于网格尺寸的正数。"

    # restOffset
    if name == "physxCollision:restOffset":
        try:
            v = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_str)[0])
        except Exception:
            v = None
        if v is not None:
            if v < 0:
                note += " restOffset < 0，一般不推荐负值。"

    # 关节刚度（特别大的）
    if name.endswith(":physics:stiffness"):
        try:
            v = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_str)[0])
        except Exception:
            v = None
        if v is not None and v > 1e6:
            note += " 刚度非常大，可能导致关节发抖或不稳定。"

    # centerOfMass = (-inf, -inf, -inf)
    if name == "physics:centerOfMass" and "-inf" in value_str:
        note += " 这是自动计算质心的占位值，不一定是错误。"

    return note


def filter_dump(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        header_done = False
        buffer_block = []
        block_has_physics = False

        for line in f_in:
            # 先把一开始的 Stage Info 原封不动写出去
            if not header_done:
                f_out.write(line)
                if line.startswith("Prim:"):
                    header_done = True
                    buffer_block = [line]
                continue

            # 新的 Prim 开始
            if line.lstrip().startswith("Prim: "):
                # 把上一个 block 写出去（如果里面有物理信息）
                if buffer_block and block_has_physics:
                    f_out.writelines(buffer_block)
                    f_out.write("\n")
                buffer_block = [line]
                block_has_physics = False
                continue

            # 处理当前 Prim 内的行
            stripped = line.strip()

            if stripped.startswith("attr "):
                # 解析属性名
                m = re.match(r"\s*attr\s+([^ ]+)", line)
                if m:
                    name = m.group(1)
                    if is_interesting_attr_name(name):
                        # 这个属性比较重要，保留 + 可能加注释
                        note = warning_for_attr(name, stripped)
                        if note:
                            base = line.rstrip("\n")
                            new_line = base + "    # " + note + "\n"
                        else:
                            new_line = line
                        buffer_block.append(new_line)
                        block_has_physics = True
                    else:
                        # 非物理相关属性就丢掉
                        pass
                else:
                    # 没识别出名字，保险起见保留
                    buffer_block.append(line)

            elif stripped.startswith("rel "):
                # 只保留 physics 相关的 relationship
                if "physics:" in stripped or "material:binding:physics" in stripped:
                    buffer_block.append(line)
                    block_has_physics = True
            else:
                # 其他行（空行、worldBBoxSize 等）：
                # 如果这个 block 最终要保留，就一起保留
                buffer_block.append(line)

        # 最后一个 block
        if buffer_block and block_has_physics:
            f_out.writelines(buffer_block)


if __name__ == "__main__":
    print("过滤输入文件:", INPUT_PATH)
    filter_dump(INPUT_PATH, OUTPUT_PATH)
    print("已生成精简版:", OUTPUT_PATH)
