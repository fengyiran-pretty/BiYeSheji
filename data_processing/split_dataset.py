"""
数据集划分模块
Dataset splitting module for train/validation/test sets
"""

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        stratify: 是否进行分层抽样
        random_state: 随机种子
        
    Returns:
        训练集、验证集、测试集文件列表的元组
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须等于 1.0"
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 收集所有图像文件和标签
    image_files = []
    labels = []
    
    # 假设数据按类别组织在子目录中
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # 创建对应的输出目录
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # 收集该类别的所有图像
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(class_path, img_file))
                labels.append(class_name)
    
    logger.info(f"共找到 {len(image_files)} 张图像")
    
    # 第一次划分：分离出测试集
    if stratify:
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            image_files, labels,
            test_size=test_ratio,
            stratify=labels,
            random_state=random_state
        )
    else:
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            image_files, labels,
            test_size=test_ratio,
            random_state=random_state
        )
    
    # 第二次划分：从剩余数据中分离出验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    if stratify:
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels,
            random_state=random_state
        )
    else:
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels,
            test_size=val_ratio_adjusted,
            random_state=random_state
        )
    
    # 复制文件到相应目录
    logger.info("复制文件到训练集...")
    copy_files(train_files, train_labels, train_dir)
    
    logger.info("复制文件到验证集...")
    copy_files(val_files, val_labels, val_dir)
    
    logger.info("复制文件到测试集...")
    copy_files(test_files, test_labels, test_dir)
    
    # 打印统计信息
    logger.info(f"\n数据集划分完成:")
    logger.info(f"  训练集: {len(train_files)} 张图像 ({len(train_files)/len(image_files)*100:.1f}%)")
    logger.info(f"  验证集: {len(val_files)} 张图像 ({len(val_files)/len(image_files)*100:.1f}%)")
    logger.info(f"  测试集: {len(test_files)} 张图像 ({len(test_files)/len(image_files)*100:.1f}%)")
    
    return train_files, val_files, test_files


def copy_files(file_list: List[str], label_list: List[str], output_dir: str) -> None:
    """
    复制文件到目标目录
    
    Args:
        file_list: 文件路径列表
        label_list: 标签列表
        output_dir: 输出目录
    """
    for file_path, label in zip(file_list, label_list):
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_dir, label, filename)
        shutil.copy2(file_path, dest_path)


def split_dataset_from_csv(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    image_col: str = 'image',
    label_col: str = 'diagnosis',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_state: int = 42
) -> pd.DataFrame:
    """
    从 CSV 文件读取数据集信息并进行划分
    
    Args:
        csv_path: CSV 文件路径
        image_dir: 图像目录
        output_dir: 输出目录
        image_col: 图像文件名列
        label_col: 标签列
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        stratify: 是否进行分层抽样
        random_state: 随机种子
        
    Returns:
        包含划分信息的 DataFrame
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)
    logger.info(f"从 CSV 读取 {len(df)} 条记录")
    
    # 第一次划分：分离测试集
    if stratify and label_col in df.columns:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            stratify=df[label_col],
            random_state=random_state
        )
    else:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=random_state
        )
    
    # 第二次划分：分离验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    if stratify and label_col in train_val_df.columns:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            stratify=train_val_df[label_col],
            random_state=random_state
        )
    else:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=random_state
        )
    
    # 添加划分标记
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    test_df['split'] = 'test'
    
    # 合并并保存
    result_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    output_csv = os.path.join(output_dir, 'dataset_split.csv')
    result_df.to_csv(output_csv, index=False)
    
    logger.info(f"数据集划分信息已保存到: {output_csv}")
    logger.info(f"  训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    
    return result_df


def create_balanced_dataset(
    data_dir: str,
    output_dir: str,
    samples_per_class: Optional[int] = None,
    random_state: int = 42
) -> None:
    """
    创建类别平衡的数据集（用于处理类别不平衡问题）
    
    Args:
        data_dir: 输入数据目录
        output_dir: 输出数据目录
        samples_per_class: 每个类别的样本数（None 表示使用最小类别的样本数）
        random_state: 随机种子
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_state)
    
    # 统计每个类别的样本数
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(images)
    
    # 确定每个类别的样本数
    if samples_per_class is None:
        samples_per_class = min(class_counts.values())
    
    logger.info(f"创建平衡数据集，每个类别 {samples_per_class} 个样本")
    
    # 为每个类别随机选择样本
    for class_name, count in class_counts.items():
        class_path = os.path.join(data_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 随机选择样本
        if len(images) > samples_per_class:
            selected_images = np.random.choice(images, samples_per_class, replace=False)
        else:
            selected_images = images
        
        # 复制文件
        for img in selected_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_class_path, img)
            shutil.copy2(src, dst)
        
        logger.info(f"  {class_name}: {len(selected_images)} 个样本")
    
    logger.info("平衡数据集创建完成！")


if __name__ == "__main__":
    print("数据集划分模块已加载")
    print("使用示例:")
    print("  split_dataset('data/raw', 'data/processed', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)")
    print("  split_dataset_from_csv('data/metadata.csv', 'data/images', 'data/processed')")
