import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import tqdm

def restore_depth_image(depth_image_path, output_path, max_depth=10.0):
    """
    还原单个深度图像
    
    Args:
        depth_image_path: 输入的归一化深度图像路径
        output_path: 输出的还原深度图像路径
        max_depth: 原始深度的最大值，默认10.0米
    """
    try:
        # 读取归一化的深度图像
        normalized_depth = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
        
        if normalized_depth is None:
            print(f"无法读取图像: {depth_image_path}")
            return False
        
        # 还原深度值：从[0, 255]映射回[0, max_depth]
        restored_depth = (normalized_depth.astype(np.float32) / 255.0) * max_depth
        
        # 保存为原始深度值（浮点数格式）
        # 可以选择保存为不同格式：
        
        # 方法1: 保存为16位整数 (推荐，节省空间且精度足够)
        # 将深度值乘以1000转换为毫米，然后保存为16位整数
        depth_mm = (restored_depth * 1000).astype(np.uint16)
        cv2.imwrite(output_path.replace('.png', '_restored_mm.png'), depth_mm)
        
        # 方法2: 保存为32位浮点数 (精度最高，但文件较大)
        np.save(output_path.replace('.png', '_restored_float.npy'), restored_depth)
        
        # 方法3: 保存为可视化的深度图（彩色映射）
        depth_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        cv2.imwrite(output_path.replace('.png', '_restored_colored.png'), depth_colored)
        
        return True
    
    except Exception as e:
        print(f"处理图像 {depth_image_path} 时出错: {e}")
        return False

def batch_restore_depth_images(input_dir, output_dir=None, max_depth=10.0, recursive=True):
    """
    批量还原文件夹中的深度图像
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径，如果为None则在原文件夹创建restored子文件夹
        max_depth: 原始深度的最大值
        recursive: 是否递归处理子文件夹
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"输入文件夹不存在: {input_dir}")
        return
    
    # 设置输出文件夹
    if output_dir is None:
        output_path = input_path / "restored_depth"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有包含"depth"的图像文件
    if recursive:
        depth_files = list(input_path.rglob("*depth*.png")) + \
                     list(input_path.rglob("*depth*.jpg")) + \
                     list(input_path.rglob("*depth*.jpeg"))
    else:
        depth_files = list(input_path.glob("*depth*.png")) + \
                     list(input_path.glob("*depth*.jpg")) + \
                     list(input_path.glob("*depth*.jpeg"))
    
    if not depth_files:
        print(f"在 {input_dir} 中未找到包含'depth'的图像文件")
        return
    
    print(f"找到 {len(depth_files)} 个深度图像文件")
    
    success_count = 0
    failed_count = 0
    
    # 批量处理
    for depth_file in tqdm.tqdm(depth_files, desc="处理深度图像"):
        # 计算相对路径以保持文件夹结构
        relative_path = depth_file.relative_to(input_path)
        output_file_path = output_path / relative_path
        
        # 创建输出文件夹
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if restore_depth_image(str(depth_file), str(output_file_path), max_depth):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n处理完成!")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    print(f"还原的深度图像保存在: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="批量还原归一化的深度图像")
    parser.add_argument("input_dir", help="输入文件夹路径")
    parser.add_argument("--output_dir", "-o", help="输出文件夹路径 (默认为输入文件夹下的restored_depth)")
    parser.add_argument("--max_depth", "-m", type=float, default=10.0, help="原始深度的最大值 (默认10.0米)")
    parser.add_argument("--no_recursive", action="store_true", help="不递归处理子文件夹")
    
    args = parser.parse_args()
    
    batch_restore_depth_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_depth=args.max_depth,
        recursive=not args.no_recursive
    )

if __name__ == "__main__":
    # 如果直接运行脚本，可以修改这里的参数
    input_folder = "/path/to/your/depth/images"  # 修改为你的输入文件夹路径
    output_folder = None  # 可以指定输出文件夹，或设为None使用默认
    max_depth_value = 10.0  # 根据你的原始数据调整
    
    # 直接调用函数（用于脚本内运行）
    # batch_restore_depth_images(input_folder, output_folder, max_depth_value)
    
    # 或者使用命令行参数
    main()