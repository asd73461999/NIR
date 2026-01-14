"""NIR图像处理主程序"""

import argparse
import os
import cv2
from image_processor import InfraredEdgeDetector, enhance_image_clahe, display_comparison, save_processed_image
from batch_processor import batch_process_all_dirs, process_directory
from metrics_calculator import calculate_metrics


def main():
    parser = argparse.ArgumentParser(description='NIR图像处理工具')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'metrics'], 
                        default='single', help='处理模式: single(单张), batch(批量), metrics(评估指标)')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录路径')
    parser.add_argument('--output', type=str, help='输出图像或目录路径')
    parser.add_argument('--ref', type=str, help='参照图像路径(仅用于metrics模式)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.output:
            print("请提供输出路径")
            return
            
        if not os.path.exists(args.input):
            print(f"输入路径不存在: {args.input}")
            return
            
        print("正在处理单张图像...")
        result = enhance_image_clahe(args.input)
        if result:
            original_img, processed_img = result
            display_comparison(original_img, processed_img)
            save_processed_image(processed_img, args.input, args.output)
    
    elif args.mode == 'batch':
        if not args.output:
            print("请提供输出路径")
            return
            
        if not os.path.exists(args.input):
            print(f"输入路径不存在: {args.input}")
            return
            
        print("正在批量处理图像...")
        batch_process_all_dirs(args.input, args.output)
    
    elif args.mode == 'metrics':
        if not args.ref:
            print("请提供参照图像路径")
            return
            
        if not os.path.exists(args.ref):
            print(f"参照图像路径不存在: {args.ref}")
            return
            
        if not os.path.exists(args.input):
            print(f"对比图像目录不存在: {args.input}")
            return
            
        print("正在计算评估指标...")
        metrics_df = calculate_metrics(
            args.ref, 
            args.input,
            result_csv=os.path.join(args.output or './', 'metrics.csv')
        )
        print(metrics_df.describe())


def demo():
    """演示功能"""
    print("=== NIR图像处理演示 ===")
    
    # 示例路径，请替换为您实际的图像路径
    sample_image_path = r"C:\Users\1\BaiduSyncdisk\NIR\NIR\5\_MG_7703.JPG"
    
    if os.path.exists(sample_image_path):
        # 初始化检测器
        detector = InfraredEdgeDetector(sigma=1.5)
        
        try:
            # 检测边缘（带可视化）
            edges = detector.detect_edges(sample_image_path, visualize=True)
            print("边缘检测完成")
        except Exception as e:
            print(f"处理过程中出现错误: {str(e)}")
    else:
        print(f"示例图像路径不存在: {sample_image_path}")
        print("请确保有可用的图像文件后再运行")


if __name__ == "__main__":
    # 如果没有命令行参数，则运行演示
    import sys
    if len(sys.argv) == 1:
        demo()
    else:
        main()