"""
Grad-CAM 可视化模块
Gradient-weighted Class Activation Mapping for model interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM 实现类"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        初始化 Grad-CAM
        
        Args:
            model: PyTorch 模型
            target_layer: 目标卷积层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.hook_layers()
    
    def hook_layers(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        生成 Grad-CAM 热力图
        
        Args:
            input_tensor: 输入图像张量 (1, C, H, W)
            target_class: 目标类别索引（None 表示使用预测类别）
            
        Returns:
            热力图 (H, W)
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # 获取梯度和激活
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)
        
        # 计算权重（全局平均池化）
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # 加权求和
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = np.maximum(cam, 0)
        
        # 归一化到 [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def visualize(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        生成叠加了热力图的可视化图像
        
        Args:
            input_tensor: 输入图像张量
            original_image: 原始图像 (H, W, 3) RGB 格式
            target_class: 目标类别
            alpha: 热力图透明度
            colormap: OpenCV 颜色映射
            
        Returns:
            可视化图像 (H, W, 3)
        """
        # 生成 CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # 调整 CAM 大小以匹配原始图像
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 归一化原始图像
        if original_image.max() > 1:
            original_image = original_image.astype(np.float32) / 255.0
        
        # 叠加
        visualization = heatmap.astype(np.float32) / 255.0
        visualization = alpha * visualization + (1 - alpha) * original_image
        visualization = np.clip(visualization, 0, 1)
        
        return visualization


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ 增强版本"""
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        生成 Grad-CAM++ 热力图
        
        Args:
            input_tensor: 输入图像张量
            target_class: 目标类别
            
        Returns:
            热力图
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward(retain_graph=True)
        
        # 获取梯度和激活
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # 计算 alpha 权重
        numerator = gradients ** 2
        denominator = 2 * (gradients ** 2) + \
                     np.sum(activations * (gradients ** 3), axis=(1, 2), keepdims=True)
        
        alpha = numerator / (denominator + 1e-7)
        
        # 计算权重
        weights = np.sum(alpha * np.maximum(gradients, 0), axis=(1, 2))
        
        # 加权求和
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = np.maximum(cam, 0)
        
        # 归一化
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


def get_target_layer(model: nn.Module, model_type: str = 'resnet') -> nn.Module:
    """
    自动获取目标层
    
    Args:
        model: PyTorch 模型
        model_type: 模型类型
        
    Returns:
        目标卷积层
    """
    if model_type.lower().startswith('resnet'):
        # ResNet: 最后一个残差块
        if hasattr(model, 'backbone'):
            return model.backbone.layer4[-1]
        else:
            return model.layer4[-1]
    
    elif model_type.lower().startswith('efficientnet'):
        # EfficientNet: 最后一个卷积块
        if hasattr(model, 'backbone'):
            return model.backbone.conv_head
        else:
            return model.conv_head
    
    elif model_type.lower().startswith('densenet'):
        # DenseNet: 最后一个 dense block
        if hasattr(model, 'backbone'):
            return model.backbone.features[-1]
        else:
            return model.features[-1]
    
    else:
        logger.warning(f"未知模型类型: {model_type}，请手动指定目标层")
        return None


def visualize_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_names: List[str],
    target_layer: nn.Module,
    save_path: Optional[str] = None,
    use_gradcam_pp: bool = False
):
    """
    完整的 Grad-CAM 可视化流程
    
    Args:
        model: PyTorch 模型
        image_tensor: 输入图像张量
        original_image: 原始图像
        class_names: 类别名称列表
        target_layer: 目标层
        save_path: 保存路径
        use_gradcam_pp: 是否使用 Grad-CAM++
    """
    # 创建 Grad-CAM 对象
    if use_gradcam_pp:
        grad_cam = GradCAMPlusPlus(model, target_layer)
    else:
        grad_cam = GradCAM(model, target_layer)
    
    # 获取预测
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[predicted_class].item()
    
    # 生成可视化
    visualization = grad_cam.visualize(image_tensor, original_image)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 热力图
    cam = grad_cam.generate_cam(image_tensor)
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM 热力图')
    axes[1].axis('off')
    
    # 叠加图像
    axes[2].imshow(visualization)
    title = f'预测: {class_names[predicted_class]}\n置信度: {confidence:.2%}'
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Grad-CAM 可视化模块已加载")
    print("使用示例:")
    print("  from visualization.grad_cam import GradCAM, get_target_layer, visualize_gradcam")
    print("  ")
    print("  target_layer = get_target_layer(model, 'resnet50')")
    print("  grad_cam = GradCAM(model, target_layer)")
    print("  cam = grad_cam.generate_cam(image_tensor)")
    print("  visualization = grad_cam.visualize(image_tensor, original_image)")
