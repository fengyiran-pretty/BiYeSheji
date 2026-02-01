"""
模型训练模块
Model training module for skin lesion classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Optional, Tuple
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinLesionDataset(Dataset):
    """皮肤病变数据集类"""
    
    def __init__(self, root_dir: str, transform=None):
        """
        初始化数据集
        
        Args:
            root_dir: 数据根目录
            transform: 数据增强/转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root_dir)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform:
            # 转换 PIL 图像为 numpy 数组
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 50,
        save_dir: str = 'checkpoints',
        early_stopping_patience: int = 10
    ):
        """
        初始化训练器
        
        Args:
            model: PyTorch 模型
            device: 训练设备
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            early_stopping_patience: 早停耐心值
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.early_stopping_patience = early_stopping_patience
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # 早停变量
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个 epoch
        
        Returns:
            平均损失和准确率
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """
        验证模型
        
        Returns:
            平均损失和准确率
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self) -> Dict:
        """
        执行完整训练流程
        
        Returns:
            训练历史字典
        """
        logger.info(f"开始训练，共 {self.num_epochs} 个 epoch")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            logger.info("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # 打印结果
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_loss, val_acc)
                logger.info(f"保存最佳模型 (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"\n早停触发！最佳 epoch: {self.best_epoch}")
                break
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_loss, val_acc)
        
        # 保存训练历史
        self.save_history()
        
        logger.info("\n训练完成！")
        logger.info(f"最佳模型: epoch {self.best_epoch}, val_loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float, val_acc: float):
        """
        保存模型检查点
        
        Args:
            filename: 文件名
            epoch: 当前 epoch
            val_loss: 验证损失
            val_acc: 验证准确率
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
    
    def save_history(self):
        """保存训练历史到 JSON 文件"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"训练历史已保存到: {history_path}")


def create_data_loaders(
    train_dir: str,
    val_dir: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        train_dir: 训练集目录
        val_dir: 验证集目录
        train_transform: 训练集数据增强
        val_transform: 验证集数据增强
        batch_size: 批次大小
        num_workers: 工作线程数
        
    Returns:
        训练和验证数据加载器
    """
    train_dataset = SkinLesionDataset(train_dir, transform=train_transform)
    val_dataset = SkinLesionDataset(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("模型训练模块已加载")
    print("使用示例:")
    print("  from models.cnn_model import get_model")
    print("  from data_processing.preprocess import get_training_augmentation, get_validation_augmentation")
    print("  ")
    print("  model = get_model('resnet50', num_classes=2)")
    print("  train_transform = get_training_augmentation()")
    print("  val_transform = get_validation_augmentation()")
    print("  train_loader, val_loader = create_data_loaders('data/train', 'data/val', train_transform, val_transform)")
    print("  ")
    print("  trainer = Trainer(model, device, train_loader, val_loader, criterion, optimizer)")
    print("  history = trainer.train()")
