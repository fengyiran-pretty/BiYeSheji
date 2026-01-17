"""
数据预处理和增强模块
Data preprocessing and augmentation module for skin lesion images
"""

import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinLesionPreprocessor:
    """皮肤病变图像预处理器"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        初始化预处理器
        
        Args:
            image_size: 目标图像尺寸 (高, 宽)
        """
        self.image_size = image_size
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        标准化图像
        
        Args:
            image: 输入图像 (H, W, C)
            
        Returns:
            标准化后的图像
        """
        # 转换为浮点数 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # 使用 ImageNet 均值和标准差
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = (image - mean) / std
        return image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            
        Returns:
            调整大小后的图像
        """
        return cv2.resize(image, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
    
    def remove_hair(self, image: np.ndarray) -> np.ndarray:
        """
        去除图像中的毛发伪影（可选预处理步骤）
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 使用黑帽变换检测深色细线（毛发）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # 二值化
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # 修复
        result = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
        
        return result


def get_training_augmentation(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """
    获取训练数据增强管道
    
    Args:
        image_size: 目标图像尺寸
        
    Returns:
        Albumentations 增强管道
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
            A.MotionBlur(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(p=1.0),
            A.GridDistortion(p=1.0),
            A.ElasticTransform(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_validation_augmentation(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """
    获取验证/测试数据增强管道（仅调整大小和标准化）
    
    Args:
        image_size: 目标图像尺寸
        
    Returns:
        Albumentations 增强管道
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def load_and_preprocess_image(
    image_path: str,
    transform: Optional[A.Compose] = None,
    remove_hair: bool = False
) -> np.ndarray:
    """
    加载并预处理单张图像
    
    Args:
        image_path: 图像路径
        transform: Albumentations 增强管道
        remove_hair: 是否去除毛发
        
    Returns:
        预处理后的图像张量
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    # 转换为 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 去除毛发（可选）
    if remove_hair:
        preprocessor = SkinLesionPreprocessor()
        image = preprocessor.remove_hair(image)
    
    # 应用数据增强
    if transform:
        augmented = transform(image=image)
        image = augmented['image']
    
    return image


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    remove_hair: bool = False
) -> None:
    """
    批量预处理数据集
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        image_size: 目标图像尺寸
        remove_hair: 是否去除毛发
    """
    os.makedirs(output_dir, exist_ok=True)
    preprocessor = SkinLesionPreprocessor(image_size)
    
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    logger.info(f"处理 {len(image_files)} 张图像...")
    
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        try:
            # 加载图像
            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 去除毛发
            if remove_hair:
                image = preprocessor.remove_hair(image)
            
            # 调整大小
            image = preprocessor.resize_image(image)
            
            # 保存
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image_bgr)
            
        except Exception as e:
            logger.error(f"处理图像 {img_file} 时出错: {e}")
    
    logger.info("预处理完成！")


if __name__ == "__main__":
    # 示例用法
    print("数据预处理模块已加载")
    print("使用示例:")
    print("  train_transform = get_training_augmentation()")
    print("  val_transform = get_validation_augmentation()")
    print("  image = load_and_preprocess_image('path/to/image.jpg', train_transform)")
