#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的Web演示 - 不需要PyTorch依赖
Simplified web demo without PyTorch dependencies
"""

import gradio as gr
import numpy as np
import random

def classify_image_simple(image):
    """简单的图像分类演示（不需要真实模型）"""
    
    if image is None:
        return {"错误": 1.0}, np.zeros((224, 224, 3), dtype=np.uint8)
    
    # 生成随机的分类结果（演示用）
    benign_prob = random.uniform(0.3, 0.9)
    malignant_prob = 1.0 - benign_prob
    
    confidences = {
        "良性 (Benign)": float(benign_prob),
        "恶性 (Malignant)": float(malignant_prob)
    }
    
    # 创建简单的伪热力图
    if len(image.shape) == 2:
        # 灰度图转RGB
        image = np.stack([image] * 3, axis=-1)
    
    h, w = image.shape[:2]
    
    # 创建径向渐变作为演示热力图
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


def create_demo_interface():
    """创建演示界面"""
    
    title = "🔬 皮肤病变智能分类系统 [简化演示]"
    
    description = """
    ## 智能皮肤癌检测与筛查系统 - 界面演示
    
    ⚠️ **注意**: 这是**简化演示版本**，不需要安装深度学习框架。
    
    - 分类结果是**随机生成的**，仅用于展示界面功能
    - 热力图是**模拟的**，不是真实的Grad-CAM输出
    - 此演示用于展示**界面布局和交互流程**
    
    ### 使用方法:
    1. 上传任意图像（建议使用皮肤图像）
    2. 系统将显示模拟的分类结果
    3. 查看模拟的热力图可视化
    
    ### 安装完整版本:
    ```bash
    # 安装所有依赖
    pip install -r requirements.txt
    
    # 运行完整版应用
    python web_app/app.py --model_path checkpoints/best_model.pth
    ```
    """
    
    article = """
    ### 技术说明
    
    **本演示版 vs 完整版**:
    
    | 功能 | 本演示版 | 完整版 |
    |------|---------|--------|
    | 依赖 | 仅Gradio+NumPy | PyTorch+timm+Albumentations |
    | 模型 | 无 | ResNet/EfficientNet/DenseNet |
    | 预测 | 随机生成 | 真实深度学习模型 (AUC>0.85) |
    | 热力图 | 简单渐变 | 真实Grad-CAM输出 |
    | 用途 | 界面演示 | 实际诊断辅助 |
    
    ### 项目信息
    - **框架**: PyTorch + Gradio
    - **模型**: ResNet/EfficientNet/DenseNet
    - **数据集**: ISIC皮肤病变图像数据集
    - **可视化**: Grad-CAM (梯度加权类激活映射)
    
    本项目作为毕业设计开发，展示了深度学习在医学图像分析中的应用。
    """
    
    interface = gr.Interface(
        fn=classify_image_simple,
        inputs=gr.Image(label="上传图像（任意图像都可用于演示）"),
        outputs=[
            gr.Label(num_top_classes=5, label="分类结果与置信度（演示结果）"),
            gr.Image(label="热力图可视化（模拟效果）")
        ],
        title=title,
        description=description,
        article=article
    )
    
    return interface


if __name__ == "__main__":
    print("="*70)
    print("🔬 皮肤病变智能分类系统 - 简化演示版")
    print("="*70)
    print()
    print("✨ 这是简化演示版，不需要安装PyTorch等深度学习框架")
    print("   仅展示界面功能，分类结果为随机生成")
    print()
    print("📝 要使用真实模型，请:")
    print("   1. 安装依赖: pip install -r requirements.txt")
    print("   2. 运行: python web_app/app.py --model_path checkpoints/best_model.pth")
    print()
    
    interface = create_demo_interface()
    
    print("\n" + "="*70)
    print("🚀 Web界面启动中...")
    print("="*70)
    print("\n📍 访问地址: http://localhost:7860")
    print("\n⏹️  按 Ctrl+C 停止服务器")
    print("="*70 + "\n")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # 创建公共链接
    )
