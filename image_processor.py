"""红外图像处理工具类"""

import cv2
import numpy as np
import os


class InfraredEdgeDetector:
    """红外图像边缘检测器"""
    
    def __init__(self, radius=4, eps=1e-6, ssim_thresh=0.9, sigma=1.5):
        """
        初始化参数
        :param radius: 引导滤波半径
        :param eps: 引导滤波epsilon值
        :param ssim_thresh: SSIM阈值
        :param sigma: 高斯核参数
        """
        self.radius = radius
        self.eps = eps
        self.ssim_thresh = ssim_thresh
        self.sigma = sigma

    def preprocess_infrared(self, image_path):
        """
        预处理红外图像
        :param image_path: 图像路径
        :return: 预处理后的图像
        """
        if not os.path.isfile(image_path):
            raise ValueError("Invalid image path provided.")
        
        img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError("Image not found or invalid path")
            
        img_min, img_max = img.min(), img.max()
        if img_max - img_min == 0:
            img_norm = np.zeros_like(img, dtype=np.uint8)
        else:
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        filtered = cv2.bilateralFilter(img_norm, d=21, sigmaColor=10, sigmaSpace=10)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_apply = clahe.apply(filtered)
        return clahe_apply

    def calculate_spatial_moments(self, img):
        """
        计算空间矩
        :param img: 输入图像
        :return: 矩阵
        """
        moments = cv2.moments(img, binaryImage=False)
        if moments['m00'] == 0:
            raise ValueError("Zero moment detected, cannot compute center of mass.")
        x_bar = moments['m10'] / moments['m00']
        y_bar = moments['m01'] / moments['m00']
        
        M, N = img.shape
        Y, X = np.meshgrid(np.arange(N), np.arange(M))
        X_centered = X - x_bar
        Y_centered = Y - y_bar

        mu = np.zeros((3, 3))
        for p in range(3):
            for q in range(3):
                if p + q >= 2:
                    mu[p, q] = np.sum((X_centered ** p) * (Y_centered ** q) * img)
        return mu

    def detect_salient_regions(self, img):
        """
        检测显著区域
        :param img: 输入图像
        :return: 显著性掩码
        """
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_xx = cv2.Sobel(sobel_x, cv2.CV_64F, 1, 0, ksize=3)
        sobel_xy = cv2.Sobel(sobel_x, cv2.CV_64F, 0, 1, ksize=3)
        sobel_yy = cv2.Sobel(sobel_y, cv2.CV_64F, 0, 1, ksize=3)
        
        kernel = np.ones((3,3), np.float32) / 9
        H_xx = cv2.filter2D(sobel_xx, -1, kernel)
        H_xy = cv2.filter2D(sobel_xy, -1, kernel)
        H_yy = cv2.filter2D(sobel_yy, -1, kernel)

        det_H = H_xx * H_yy - H_xy**2
        trace_H = H_xx + H_yy
        discriminant = np.sqrt(np.maximum(trace_H**2 - 4*det_H, 0))
        lambda1 = 0.5 * (trace_H + discriminant)
        lambda2 = 0.5 * (trace_H - discriminant)
        
        saliency = np.sqrt(lambda1**2 + lambda2**2)
        saliency_norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        _, saliency_mask = cv2.threshold(saliency_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        saliency_mask = cv2.bitwise_not(saliency_mask)
        return saliency_mask

    def guided_filter_enhance(self, img, saliency_mask):
        """
        使用引导滤波进行增强
        :param img: 输入图像
        :param saliency_mask: 显著性掩码
        :return: 增强后的图像
        """
        guide = cv2.bitwise_and(img, img, mask=saliency_mask)
        if np.all(guide == 0):
            raise ValueError("Saliency mask is all black, cannot perform guided filtering.")
        
        mean_I = cv2.boxFilter(guide, cv2.CV_64F, (self.radius, self.radius))
        mean_p = cv2.boxFilter(img.astype(np.float64), cv2.CV_64F, (self.radius, self.radius))
        corr_I = cv2.boxFilter(guide*guide, cv2.CV_64F, (self.radius, self.radius))
        corr_Ip = cv2.boxFilter(guide*img.astype(np.float64), cv2.CV_64F, (self.radius, self.radius))
        
        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
        
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (self.radius, self.radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (self.radius, self.radius))
        
        return np.clip(mean_a * guide + mean_b, 0, 255).astype(np.uint8)

    def calculate_gray_difference(self, img, c1, c2):
        """
        计算灰度差异
        :param img: 输入图像
        :param c1: 第一个点坐标
        :param c2: 第二个点坐标
        :return: 灰度差异
        """
        x1, y1 = c1
        x2, y2 = c2

        if not (0 <= x1 < img.shape[0] and 0 <= y1 < img.shape[1]):
            raise IndexError("Point c1 is out of image bounds.")
        if not (0 <= x2 < img.shape[0] and 0 <= y2 < img.shape[1]):
            raise IndexError("Point c2 is out of image bounds.")

        kx = img[x1, y1] - img[x1-1, y1] if x1 > 0 else 0
        ky = img[x1, y1] - img[x1, y1-1] if y1 > 0 else 0
        k = np.array([kx, ky])
        
        f = img[x1, y1] - img[x2, y2]
        g = np.dot(k, k) * f + np.random.normal(0, 0.1)
        return g

    def detect_edges(self, image_path, visualize=False):
        """
        检测边缘
        :param image_path: 图像路径
        :param visualize: 是否可视化
        :return: 增强后的图像
        """
        oring_img = cv2.imread(image_path)
        preprocessed = self.preprocess_infrared(image_path)
        saliency_mask = self.detect_salient_regions(preprocessed)
        enhanced = self.guided_filter_enhance(preprocessed, saliency_mask)
        edges = cv2.Canny(enhanced, 90, 100, L2gradient=True)
        
        if visualize:
            cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Preprocessed', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Saliency Mask', cv2.WINDOW_NORMAL)
            
            cv2.moveWindow('Original', 50, 50)
            cv2.moveWindow('Preprocessed', 50,600)
            cv2.moveWindow('Saliency Mask', 700, 50)
            
            cv2.imshow('Original', oring_img)
            cv2.imshow('Preprocessed', preprocessed)
            cv2.imshow('Saliency Mask', saliency_mask)
            
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                    
            cv2.destroyAllWindows()

        self.preprocessed = preprocessed
        self.saliency_mask = saliency_mask
        self.enhanced = enhanced

        return enhanced


def sharpen_image(image):
    """
    锐化图像
    :param image: 输入图像
    :return: 锐化后的图像
    """
    kernel = np.array([[-1,-1,-1], 
                      [-1, 9.05,-1],
                      [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)


def enhance_image_clahe(input_path):
    """
    使用CLAHE增强图像
    :param input_path: 输入图像路径
    :return: 原始图像和处理后图像
    """
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} does not exist")
        return None
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Warning: Unable to read file {input_path}")
        return None
    
    clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(3,3))
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    clahe_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    sharpened = sharpen_image(clahe_img)
    
    return img, sharpened


def display_comparison(original, processed, original_title="Original", processed_title="Processed"):
    """
    显示图像对比
    :param original: 原始图像
    :param processed: 处理后图像
    :param original_title: 原始图像标题
    :param processed_title: 处理后图像标题
    """
    original_resized = cv2.resize(original, (600, 400))
    processed_resized = cv2.resize(processed, (600, 400))
    
    cv2.namedWindow(original_title, cv2.WINDOW_NORMAL)
    cv2.namedWindow(processed_title, cv2.WINDOW_NORMAL)
    
    cv2.moveWindow(original_title, 50, 50)
    cv2.moveWindow(processed_title, 700, 50)
    
    cv2.imshow(original_title, original_resized)
    cv2.imshow(processed_title, processed_resized)
    
    print("Press ESC to close the windows")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()


def save_processed_image(processed_img, input_path, output_dir="./output"):
    """
    保存处理后的图像
    :param processed_img: 处理后的图像
    :param input_path: 输入图像路径
    :param output_dir: 输出目录
    :return: 保存是否成功
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_processed{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    success = cv2.imwrite(output_path, processed_img)
    if success:
        print(f"Processed image saved to: {output_path}")
    else:
        print(f"Failed to save processed image to: {output_path}")
    
    return success