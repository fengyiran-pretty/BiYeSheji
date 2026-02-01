#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持久运行的Web服务器 - 用于在受限环境中运行
Persistent web server for restricted environments
"""

import gradio as gr
import numpy as np
import random
import sys
import signal
import time

# 全局标志用于优雅关闭
keep_running = True

def signal_handler(sig, frame):
    global keep_running
    print('\n\n正在关闭服务器...')
    keep_running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
    
    title = "🔬 皮肤病变智能分类系统 [演示版]"
    
    description = """
    ## 智能皮肤癌检测与筛查系统
    
    ⚠️ **注意**: 这是演示版本
    
    - ✅ 界面功能完整可用
    - ❌ 分类结果为随机生成（仅用于演示）
    - ❌ 热力图为模拟效果
    
    ### 使用方法:
    1. 上传任意图像
    2. 点击Submit按钮
    3. 查看演示结果
    """
    
    article = """
    ### 技术栈
    - **框架**: Gradio + NumPy
    - **完整版**: 需要PyTorch、timm、Albumentations
    - **模型**: ResNet/EfficientNet (完整版)
    - **数据集**: ISIC皮肤病变数据集
    """
    
    interface = gr.Interface(
        fn=classify_image_simple,
        inputs=gr.Image(label="上传图像"),
        outputs=[
            gr.Label(num_top_classes=5, label="分类结果（演示）"),
            gr.Image(label="热力图（模拟）")
        ],
        title=title,
        description=description,
        article=article
    )
    
    return interface


if __name__ == "__main__":
    print("="*70)
    print("🔬 皮肤病变智能分类系统 - 持久服务器")
    print("="*70)
    print()
    print("✨ 这是演示版本，分类结果为随机生成")
    print()
    
    try:
        interface = create_demo_interface()
        
        print("🚀 启动Web服务器...")
        print("📍 本地访问: http://localhost:7860")
        print("⏹️  按 Ctrl+C 停止服务器")
        print("="*70)
        print()
        
        # 启动服务器，禁用share以避免连接问题
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            prevent_thread_lock=False,
            quiet=False,
            show_error=True
        )
        
        # 保持运行
        while keep_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
