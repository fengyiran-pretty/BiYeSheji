"""
CNN 模型定义
Convolutional Neural Network models for skin lesion classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicCNN(nn.Module):
    """基础 CNN 模型（适用于概念验证）"""
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        """
        初始化基础 CNN 模型
        
        Args:
            num_classes: 类别数量
            dropout: Dropout 比例
        """
        super(BasicCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, H, W)
            
        Returns:
            输出张量 (B, num_classes)
        """
        # 卷积块 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积块 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积块 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 卷积块 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResNetClassifier(nn.Module):
    """基于 ResNet 的分类器"""
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        初始化 ResNet 分类器
        
        Args:
            model_name: ResNet 模型名称 (resnet18, resnet34, resnet50, resnet101)
            num_classes: 类别数量
            pretrained: 是否使用预训练权重
            dropout: Dropout 比例
        """
        super(ResNetClassifier, self).__init__()
        
        # 加载预训练模型
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # 移除分类头
        )
        
        # 获取特征维度
        num_features = self.backbone.num_features
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"已加载 {model_name}，特征维度: {num_features}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class EfficientNetClassifier(nn.Module):
    """基于 EfficientNet 的分类器"""
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        初始化 EfficientNet 分类器
        
        Args:
            model_name: EfficientNet 模型名称 (efficientnet_b0 到 efficientnet_b7)
            num_classes: 类别数量
            pretrained: 是否使用预训练权重
            dropout: Dropout 比例
        """
        super(EfficientNetClassifier, self).__init__()
        
        # 加载预训练模型
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        
        # 获取特征维度
        num_features = self.backbone.num_features
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"已加载 {model_name}，特征维度: {num_features}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class DenseNetClassifier(nn.Module):
    """基于 DenseNet 的分类器"""
    
    def __init__(
        self,
        model_name: str = 'densenet121',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        初始化 DenseNet 分类器
        
        Args:
            model_name: DenseNet 模型名称 (densenet121, densenet169, densenet201)
            num_classes: 类别数量
            pretrained: 是否使用预训练权重
            dropout: Dropout 比例
        """
        super(DenseNetClassifier, self).__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"已加载 {model_name}，特征维度: {num_features}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def get_model(
    model_type: str = 'resnet50',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5
) -> nn.Module:
    """
    获取指定类型的模型
    
    Args:
        model_type: 模型类型
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        dropout: Dropout 比例
        
    Returns:
        PyTorch 模型
    """
    model_type = model_type.lower()
    
    if model_type == 'basic_cnn':
        model = BasicCNN(num_classes=num_classes, dropout=dropout)
    elif model_type.startswith('resnet'):
        model = ResNetClassifier(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_type.startswith('efficientnet'):
        model = EfficientNetClassifier(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_type.startswith('densenet'):
        model = DenseNetClassifier(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> Dict:
    """
    获取模型摘要信息
    
    Args:
        model: PyTorch 模型
        
    Returns:
        包含模型信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


if __name__ == "__main__":
    # 测试模型创建
    print("CNN 模型模块已加载")
    print("\n可用模型:")
    print("  - basic_cnn: 基础 CNN 模型")
    print("  - resnet18/34/50/101: ResNet 系列")
    print("  - efficientnet_b0 到 b7: EfficientNet 系列")
    print("  - densenet121/169/201: DenseNet 系列")
    
    print("\n使用示例:")
    print("  model = get_model('resnet50', num_classes=2, pretrained=True)")
    print("  summary = model_summary(model)")
    
    # 创建示例模型
    model = get_model('resnet18', num_classes=2, pretrained=False)
    summary = model_summary(model)
    print(f"\nResNet18 模型统计:")
    print(f"  总参数: {summary['total_parameters']:,}")
    print(f"  可训练参数: {summary['trainable_parameters']:,}")
