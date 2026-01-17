"""
Web 应用主程序
Web application for skin lesion classification using Gradio
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import logging
from typing import Tuple, Dict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import get_model
from data_processing.preprocess import get_validation_augmentation
from visualization.grad_cam import GradCAM, get_target_layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinLesionClassifier:
    """皮肤病变分类器"""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'resnet50',
        num_classes: int = 2,
        class_names: list = None,
        device: str = 'auto'
    ):
        """
        初始化分类器
        
        Args:
            model_path: 模型权重路径
            model_type: 模型类型
            num_classes: 类别数量
            class_names: 类别名称列表
            device: 设备 ('auto', 'cpu', 'cuda')
        """
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = get_model(model_type, num_classes=num_classes, pretrained=False)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"已加载模型: {model_path}")
        else:
            logger.warning(f"模型文件不存在: {model_path}，使用未训练的模型")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 设置类别名称
        if class_names is None:
            self.class_names = [f'Class_{i}' for i in range(num_classes)]
        else:
            self.class_names = class_names
        
        # 数据预处理
        self.transform = get_validation_augmentation()
        
        # Grad-CAM
        self.target_layer = get_target_layer(self.model, model_type)
        if self.target_layer:
            self.grad_cam = GradCAM(self.model, self.target_layer)
        else:
            self.grad_cam = None
            logger.warning("无法创建 Grad-CAM，可视化功能将不可用")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W, C) RGB
            
        Returns:
            预处理后的张量 (1, C, H, W)
        """
        # 应用数据增强
        augmented = self.transform(image=image)
        tensor = augmented['image'].unsqueeze(0)
        return tensor
    
    def predict(self, image: np.ndarray) -> Tuple[str, Dict[str, float], np.ndarray]:
        """
        预测图像类别
        
        Args:
            image: 输入图像
            
        Returns:
            预测类别、置信度字典、可视化图像
        """
        # 保存原始图像用于可视化
        original_image = image.copy()
        if original_image.max() > 1:
            original_image = original_image.astype(np.float32) / 255.0
        
        # 预处理
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # 获取预测结果
        predicted_idx = outputs.argmax(dim=1).item()
        predicted_class = self.class_names[predicted_idx]
        
        # 构建置信度字典
        confidences = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        # 生成 Grad-CAM 可视化
        if self.grad_cam:
            try:
                visualization = self.grad_cam.visualize(
                    input_tensor,
                    original_image,
                    target_class=predicted_idx
                )
                visualization = (visualization * 255).astype(np.uint8)
            except Exception as e:
                logger.error(f"生成 Grad-CAM 时出错: {e}")
                visualization = original_image
                if visualization.max() <= 1:
                    visualization = (visualization * 255).astype(np.uint8)
        else:
            visualization = original_image
            if visualization.max() <= 1:
                visualization = (visualization * 255).astype(np.uint8)
        
        return predicted_class, confidences, visualization


# 全局分类器实例
classifier = None


def initialize_classifier(
    model_path: str = 'checkpoints/best_model.pth',
    model_type: str = 'resnet50',
    class_names: list = None
):
    """初始化分类器"""
    global classifier
    
    if class_names is None:
        class_names = ['良性 (Benign)', '恶性 (Malignant)']
    
    classifier = SkinLesionClassifier(
        model_path=model_path,
        model_type=model_type,
        num_classes=len(class_names),
        class_names=class_names
    )


def classify_image(image: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """
    分类图像（Gradio 接口函数）
    
    Args:
        image: 输入图像
        
    Returns:
        置信度字典和可视化图像
    """
    if classifier is None:
        return {"错误": 1.0}, np.zeros((224, 224, 3), dtype=np.uint8)
    
    try:
        predicted_class, confidences, visualization = classifier.predict(image)
        return confidences, visualization
    except Exception as e:
        logger.error(f"分类时出错: {e}")
        return {"错误": 1.0}, image


def create_gradio_interface():
    """创建 Gradio 界面"""
    
    # 界面描述
    title = "🔬 皮肤病变智能分类系统"
    description = """
    ## 智能皮肤癌检测与筛查系统
    
    本系统使用深度学习技术自动分析皮肤病变图像，辅助黑色素瘤的早期检测。
    
    ### 使用方法:
    1. 上传皮肤病变图像（支持 JPG、PNG 格式）
    2. 系统将自动分析并返回分类结果
    3. 查看置信度分数和 Grad-CAM 热力图可视化
    
    ### 注意事项:
    - 本系统仅供辅助参考，不能替代专业医生诊断
    - 如有疑虑，请及时就医咨询专业皮肤科医生
    - 图像质量会影响检测准确性，建议上传清晰的病变特写照片
    """
    
    article = """
    ### 技术说明
    - **模型架构**: 基于 ResNet/EfficientNet 的卷积神经网络
    - **训练数据**: ISIC 皮肤病变图像数据集
    - **可视化方法**: Grad-CAM (梯度加权类激活映射)
    
    ### 性能指标
    - 准确率 (Accuracy): >85%
    - AUC: >0.85
    - 敏感度 (Sensitivity): 高灵敏度检测恶性病变
    
    ### 开发团队
    本项目作为毕业设计开发，使用 PyTorch 深度学习框架实现。
    """
    
    # 创建界面
    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="numpy", label="上传皮肤病变图像"),
        outputs=[
            gr.Label(num_top_classes=5, label="分类结果与置信度"),
            gr.Image(type="numpy", label="Grad-CAM 可视化")
        ],
        title=title,
        description=description,
        article=article,
        examples=[
            # 可以添加示例图像路径
        ],
        allow_flagging="never",
        analytics_enabled=False
    )
    
    return interface


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='皮肤病变分类 Web 应用')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        help='模型类型')
    parser.add_argument('--port', type=int, default=7860,
                        help='端口号')
    parser.add_argument('--share', action='store_true',
                        help='创建公共链接')
    
    args = parser.parse_args()
    
    # 初始化分类器
    logger.info("初始化分类器...")
    initialize_classifier(
        model_path=args.model_path,
        model_type=args.model_type
    )
    
    # 创建并启动界面
    logger.info("启动 Web 界面...")
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
