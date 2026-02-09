import numpy as np
import pathlib
import os

# ================= 配置区域 =================
# 1. 输入文件路径 (你原来的大文件)
source_file = '/home/yhy/IsaacLabExtensionTemplate/trajectories_reisfirst.npz'

# 2. 输出文件夹路径 (会自动创建)
# 建议放在一个新的子文件夹里，否则你的目录下会瞬间多出500个文件
output_dir = '/home/yhy/IsaacLabExtensionTemplate/data_episodes'
# ===========================================

def convert_and_save():
    # 转换路径对象
    src = pathlib.Path(source_file)
    dst = pathlib.Path(output_dir)
    
    # 创建输出目录 (如果不存在)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"📂 输出目录已准备: {dst}")
    print(f"📖 正在读取源文件: {src} ...")

    try:
        # 加载原始大文件
        raw_data = np.load(src, allow_pickle=True)
        keys = raw_data.files
        total_files = len(keys)
        print(f"🔍 发现 {total_files} 条轨迹，开始拆分保存...")

        saved_count = 0
        
        for i, uuid in enumerate(keys):
            # 1. 获取原始数据
            value = raw_data[uuid]
            
            # 2. 解包 (去除 0维 array 壳)
            if isinstance(value, np.ndarray) and value.ndim == 0:
                episode = value.item()
            else:
                episode = value
            
            # 3. 获取轨迹长度 (通常检查 'reward' 或 'action' 的长度)
            # 这里的逻辑是：尝试找 reward，找不到就找第一个键的长度
            if 'reward' in episode:
                length = len(episode['reward'])
            else:
                # 兜底：获取字典里任意一个 value 的长度
                first_key = next(iter(episode))
                length = len(episode[first_key])

            # 4. 构建文件名: {UUID}-{Length}.npz
            # 例如: dbfe1749...-1000.npz
            save_name = f"{uuid}-{length}.npz"
            save_path = dst / save_name

            # 5. 保存为压缩的独立 npz
            # **episode 的作用是把字典打散，保存成 npz 内部的多个数组
            np.savez_compressed(save_path, **episode)
            
            saved_count += 1
            
            # 每处理 50 个打印一次进度
            if (i + 1) % 50 == 0:
                print(f"   已保存 {i + 1}/{total_files} 个文件...")

        print("-" * 30)
        print(f"✅ 成功完成！")
        print(f"共保存了 {saved_count} 个独立文件到: {dst}")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    convert_and_save()