"""图像质量评估指标计算工具"""

import pandas as pd
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def calculate_metrics(ref_path, cmp_dir, result_csv="metrics.csv"):
    """
    计算图像质量评估指标（PSNR、SSIM）
    :param ref_path: 参照图像路径
    :param cmp_dir: 对比图像目录
    :param result_csv: 结果保存CSV文件名
    :return: 包含指标结果的DataFrame
    """
    # 读取参照图
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise ValueError(f"无法读取参照图片: {ref_path}")
    
    results = []
    
    # 遍历对比目录
    for cmp_name in tqdm(os.listdir(cmp_dir)):
        if not any(cmp_name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif']):
            continue
            
        cmp_path = os.path.join(cmp_dir, cmp_name)
        
        try:
            cmp = cv2.imread(cmp_path, cv2.IMREAD_GRAYSCALE)
            if cmp is None:
                print(f"无法读取对比图: {cmp_name}")
                continue
                
            # 尺寸对齐
            if ref.shape != cmp.shape:
                cmp = cv2.resize(cmp, (ref.shape[1], ref.shape[0]))
            
            # 计算PSNR
            psnr = cv2.PSNR(ref, cmp)
            
            # 计算SSIM
            ssim_score = ssim(ref, cmp, data_range=255)
            
            results.append({
                "参照文件": os.path.basename(ref_path),
                "对比文件": cmp_name,
                "PSNR": round(psnr, 2),
                "SSIM": round(ssim_score, 4)
            })
            
        except Exception as e:
            print(f"处理 {cmp_name} 出错: {str(e)}")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(result_csv, index=False)
    print(f"指标计算结果已保存至: {result_csv}")
    return df


def calculate_single_image_metrics(ref_img, cmp_img):
    """
    计算两张图像之间的PSNR和SSIM
    :param ref_img: 参照图像
    :param cmp_img: 对比图像
    :return: (PSNR, SSIM) 元组
    """
    if len(ref_img.shape) == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img
        
    if len(cmp_img.shape) == 3:
        cmp_gray = cv2.cvtColor(cmp_img, cv2.COLOR_BGR2GRAY)
    else:
        cmp_gray = cmp_img
    
    # 尺寸对齐
    if ref_gray.shape != cmp_gray.shape:
        cmp_gray = cv2.resize(cmp_gray, (ref_gray.shape[1], ref_gray.shape[0]))
    
    # 计算PSNR和SSIM
    psnr = cv2.PSNR(ref_gray, cmp_gray)
    ssim_score = ssim(ref_gray, cmp_gray, data_range=255)
    
    return psnr, ssim_score


def compare_multiple_methods(ref_path, method_results_dict, result_csv="comparison_metrics.csv"):
    """
    比较多种处理方法的结果
    :param ref_path: 参照图像路径
    :param method_results_dict: 包含不同处理方法结果的字典
    :param result_csv: 结果保存CSV文件名
    :return: 包含比较结果的DataFrame
    """
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise ValueError(f"无法读取参照图片: {ref_path}")
    
    results = []
    
    for method_name, img_path in method_results_dict.items():
        try:
            cmp = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if cmp is None:
                print(f"无法读取 {method_name} 的结果图像: {img_path}")
                continue
                
            # 尺寸对齐
            if ref.shape != cmp.shape:
                cmp = cv2.resize(cmp, (ref.shape[1], ref.shape[0]))
            
            # 计算PSNR和SSIM
            psnr = cv2.PSNR(ref, cmp)
            ssim_score = ssim(ref, cmp, data_range=255)
            
            results.append({
                "方法名称": method_name,
                "PSNR": round(psnr, 2),
                "SSIM": round(ssim_score, 4)
            })
            
        except Exception as e:
            print(f"处理 {method_name} 的结果出错: {str(e)}")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(result_csv, index=False)
    print(f"方法比较结果已保存至: {result_csv}")
    return df