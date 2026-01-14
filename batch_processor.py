"""批量图像处理工具"""

import os
import numpy as np
import cv2
from image_processor import enhance_image_clahe, InfraredEdgeDetector


def load_images_from_folder(folder):
    """
    加载文件夹中的所有图像
    :param folder: 文件夹路径
    :return: 图像路径列表
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and any(img_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif']):
            images.append(img_path)
    return images


def load_folders_from_folder(folder):
    """
    加载指定文件夹下的所有子文件夹
    :param folder: 父文件夹路径
    :return: 子文件夹路径列表
    """
    folders = []
    for foldername in os.listdir(folder):
        folder_path = os.path.join(folder, foldername)
        if os.path.isdir(folder_path):
            folders.append(folder_path)
    return folders


def process_directory(input_dir, output_dir, alpha=1.0, beta=0):
    """
    批量处理目录中的图像
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param alpha: 对比度控制参数
    :param beta: 亮度控制参数
    """
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(3,3))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有图片文件
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif']):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                img = cv2.imread(input_path)
                if img is not None:
                    # 转换为YUV颜色空间
                    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    # 对Y通道应用CLAHE
                    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
                    # 转换回BGR颜色空间
                    clahe_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    
                    # 锐化处理
                    kernel = np.array([[-1,-1,-1], 
                                      [-1, 9.05,-1],
                                      [-1,-1,-1]])
                    sharpened = cv2.filter2D(clahe_img, -1, kernel)
                    
                    # 应用对比度和亮度调整
                    adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
                    
                    # 保存处理后的图像
                    cv2.imwrite(output_path, adjusted)
                    print(f"Processed and saved: {filename}")
                else:
                    print(f"Warning: could not read file {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


def batch_process_with_detector(input_dir, output_dir):
    """
    使用边缘检测器批量处理图像
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    """
    detector = InfraredEdgeDetector(sigma=1.5)
    os.makedirs(output_dir, exist_ok=True)
    
    images = load_images_from_folder(input_dir)
    for i, image_path in enumerate(images):
        try:
            edges = detector.detect_edges(image_path)
            output_path = os.path.join(output_dir, f"edge_{i}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, edges)
            print(f"Processed edge detection for: {os.path.basename(image_path)}")
        except Exception as e:
            print(f"Error processing {image_path} for edge detection: {str(e)}")


def batch_process_all_dirs(root_input_dir, root_output_dir):
    """
    批量处理根目录下的所有子目录
    :param root_input_dir: 根输入目录
    :param root_output_dir: 根输出目录
    """
    subfolders = load_folders_from_folder(root_input_dir)
    
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        output_folder = os.path.join(root_output_dir, folder_name)
        process_directory(folder_path, output_folder)
