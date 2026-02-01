#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web应用演示脚本 - 不需要训练好的模型
Demo script for web application - works without trained model

这个脚本展示Web界面的功能，即使没有训练好的模型也可以运行。
预测结果是随机的，但可以用来演示界面布局和交互流程。
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.cnn_model import get_model
from data_processing.preprocess import get_validation_augmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoClassifier:
    """演示用分类器（不需要训练好的模型）"""
    
    def __init__(self, model_type='resnet18', num_classes=2):
        """初始化演示分类器"""
        self.device = torch.device('cpu')  # 演示模式使用CPU
        self.model_type = model_type
        self.num_classes = num_classes
        self.class_names = ['良性 (Benign)', '恶性 (Malignant)']
        
        # 创建未训练的模型
        logger.info(f"创建演示模型: {model_type}")
        self.model = get_model(model_type, num_classes=num_classes, pretrained=False)
        self.model.to(self.device)
        self.model.eval()
        
        # 数据预处理
        self.transform = get_validation_augmentation()
        
        logger.info("✅ 演示分类器初始化完成")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        augmented = self.transform(image=image)
        tensor = augmented['image'].unsqueeze(0)
        return tensor
    
    def predict(self, image: np.ndarray):
        """执行预测（演示版本）"""
        # 预处理
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # 构建置信度字典
        confidences = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        # 创建简单的伪热力图（演示用）
        # 在实际应用中，这里会使用Grad-CAM生成真实的热力图
        h, w = image.shape[:2]
        
        # 创建一个简单的径向渐变作为演示热力图
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        mask = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = 1 - (mask / mask.max())
        
        # 应用热力图颜色
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap[:, :, 0] = (mask * 255).astype(np.uint8)  # 红色通道
        
        # 叠加到原图
        if image.max() > 1:
            image_normalized = image.astype(np.float32) / 255.0
        else:
            image_normalized = image.astype(np.float32)
        
        heatmap_normalized = heatmap.astype(np.float32) / 255.0
        visualization = 0.6 * image_normalized + 0.4 * heatmap_normalized
        visualization = (np.clip(visualization, 0, 1) * 255).astype(np.uint8)
        
        return confidences, visualization


# 全局分类器实例
demo_classifier = None


def initialize_demo_classifier():
    """初始化演示分类器"""
    global demo_classifier
    demo_classifier = DemoClassifier(model_type='resnet18', num_classes=2)


def classify_image_demo(image: np.ndarray):
    """分类图像（演示版本）"""
    if demo_classifier is None:
        return {"错误": 1.0}, np.zeros((224, 224, 3), dtype=np.uint8)
    
    try:
        confidences, visualization = demo_classifier.predict(image)
        return confidences, visualization
    except Exception as e:
        logger.error(f"分类时出错: {e}")
        return {"错误": 1.0}, image


def create_demo_interface():
    """创建演示界面"""
    
    title = "🔬 皮肤病变智能分类系统 [演示模式]"
    
    description = """
    ## 智能皮肤癌检测与筛查系统 - 界面演示
    
    ⚠️ **注意**: 这是演示模式，使用的是未训练的模型。
    
    - 分类结果是**随机的**，不代表真实诊断能力
    - 热力图是**模拟的**，不是真实的Grad-CAM输出
    - 此演示仅用于展示**界面功能和交互流程**
    
    ### 使用方法:
    1. 上传任意图像（建议使用皮肤图像以获得更好的展示效果）
    2. 系统将显示模拟的分类结果
    3. 查看模拟的热力图可视化
    
    ### 实际应用:
    - 真实应用需要使用在ISIC数据集上训练的模型
    - 真实的Grad-CAM会准确显示模型关注的病变区域
    - 真实模型的AUC可以达到0.85以上
    
    ### 开始训练模型:
    ```bash
    # 1. 准备数据
    python -c "from data_processing.split_dataset import split_dataset; \\
               split_dataset('data/raw', 'data/processed')"
    
    # 2. 训练模型
    python train_model.py --model_type resnet50 --epochs 50
    
    # 3. 启动真实应用
    python web_app/app.py --model_path checkpoints/best_model.pth
    ```
    """
    
    article = """
    ### 技术说明
    
    **演示模式与真实模式的区别**:
    
    | 功能 | 演示模式 | 真实模式 |
    |------|---------|---------|
    | 模型 | 未训练（随机权重） | 在ISIC数据集上训练 |
    | 预测准确性 | 随机（50%） | AUC > 0.85 |
    | 热力图 | 简单的径向渐变 | 真实的Grad-CAM输出 |
    | 性能 | 快速 | 取决于硬件 |
    | 用途 | 界面演示 | 实际诊断辅助 |
    
    ### 真实模型性能指标
    - **准确率**: >85%
    - **AUC**: >0.85
    - **敏感度**: 高灵敏度检测恶性病变
    
    ### 项目信息
    - **框架**: PyTorch + Gradio
    - **模型**: ResNet/EfficientNet/DenseNet
    - **数据集**: ISIC皮肤病变图像数据集
    - **可视化**: Grad-CAM (梯度加权类激活映射)
    
    本项目作为毕业设计开发，展示了深度学习在医学图像分析中的应用。
    """
    
    # 创建界面
    interface = gr.Interface(
        fn=classify_image_demo,
        inputs=gr.Image(type="numpy", label="上传图像（任意图像都可用于演示）"),
        outputs=[
            gr.Label(num_top_classes=5, label="分类结果与置信度（演示结果）"),
            gr.Image(type="numpy", label="热力图可视化（模拟效果）")
        ],
        title=title,
        description=description,
        article=article,
        examples=[
            # 可以添加示例图像
        ],
        allow_flagging="never",
        analytics_enabled=False,
        theme="default"
    )
    
    return interface


def main():
    """主函数"""
    print("="*70)
    print("🔬 皮肤病变智能分类系统 - 演示模式")
    print("="*70)
    print()
    print("⚠️  这是演示模式，使用未训练的模型")
    print("   分类结果和热力图都是模拟的，仅用于展示界面功能")
    print()
    print("📝 要使用真实的训练模型，请运行:")
    print("   python web_app/app.py --model_path checkpoints/best_model.pth")
    print()
    
    # 初始化演示分类器
    logger.info("正在初始化演示分类器...")
    initialize_demo_classifier()
    
    # 创建并启动界面
    logger.info("正在创建Web界面...")
    interface = create_demo_interface()
    
    print("\n" + "="*70)
    print("🚀 演示界面已启动!")
    print("="*70)
    print("\n📍 访问地址:")
    print("   http://localhost:7860")
    print("\n💡 使用提示:")
    print("   1. 在浏览器中打开上述地址")
    print("   2. 上传任意图像进行演示")
    print("   3. 查看模拟的分类结果和可视化")
    print("\n⏹️  按 Ctrl+C 停止服务器")
    print("="*70 + "\n")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
