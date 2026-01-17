"""
模型单元测试
Unit tests for skin lesion classification models
"""

import pytest
import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import get_model, BasicCNN, ResNetClassifier, model_summary


class TestModels:
    """模型测试类"""
    
    def test_basic_cnn_creation(self):
        """测试基础 CNN 模型创建"""
        model = BasicCNN(num_classes=2)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_basic_cnn_forward(self):
        """测试基础 CNN 前向传播"""
        model = BasicCNN(num_classes=2)
        model.eval()
        
        # 创建随机输入
        x = torch.randn(1, 3, 224, 224)
        
        # 前向传播
        with torch.no_grad():
            output = model(x)
        
        # 检查输出形状
        assert output.shape == (1, 2)
    
    def test_resnet_creation(self):
        """测试 ResNet 模型创建"""
        model = get_model('resnet18', num_classes=2, pretrained=False)
        assert model is not None
    
    def test_resnet_forward(self):
        """测试 ResNet 前向传播"""
        model = get_model('resnet18', num_classes=2, pretrained=False)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 2)
    
    def test_model_summary(self):
        """测试模型摘要功能"""
        model = BasicCNN(num_classes=2)
        summary = model_summary(model)
        
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert summary['total_parameters'] > 0


class TestDataProcessing:
    """数据处理测试类"""
    
    def test_image_preprocessing(self):
        """测试图像预处理"""
        from data_processing.preprocess import get_validation_augmentation
        
        transform = get_validation_augmentation()
        
        # 创建随机图像
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 应用转换
        augmented = transform(image=image)
        tensor = augmented['image']
        
        # 检查输出形状
        assert tensor.shape == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
