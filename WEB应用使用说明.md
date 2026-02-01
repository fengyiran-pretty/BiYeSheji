# Web应用界面使用说明

## 功能概述

Web应用（`web_app/app.py`）使用Gradio框架实现了一个用户友好的皮肤病变分类界面，支持：
- 实时图像上传
- 自动分类预测
- Grad-CAM热力图可视化
- 置信度显示

## 核心实现位置

### 1. 主文件结构

```
web_app/
├── __init__.py          # 包初始化
└── app.py              # 主应用程序（约290行）
```

### 2. 核心组件实现

#### 2.1 SkinLesionClassifier 类 (第28-154行)

这是核心分类器类，封装了所有预测逻辑：

**位置**: `web_app/app.py` 第28-154行

**主要方法**:
- `__init__()`: 初始化模型、设备、Grad-CAM
- `preprocess_image()`: 图像预处理（第87-100行）
- `predict()`: 执行预测并生成可视化（第102-154行）

**关键功能**:
```python
# 第58行：加载模型
self.model = get_model(model_type, num_classes=num_classes, pretrained=False)

# 第60-63行：加载训练好的权重
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])

# 第80-82行：初始化Grad-CAM
self.target_layer = get_target_layer(self.model, model_type)
self.grad_cam = GradCAM(self.model, self.target_layer)

# 第118-123行：模型预测
input_tensor = self.preprocess_image(image).to(self.device)
with torch.no_grad():
    outputs = self.model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]

# 第136-143行：生成Grad-CAM可视化
visualization = self.grad_cam.visualize(
    input_tensor,
    original_image,
    target_class=predicted_idx
)
```

#### 2.2 Gradio接口函数

**classify_image() 函数** (第180-198行):
- 接收来自Gradio界面的图像输入
- 调用分类器的predict方法
- 返回置信度字典和可视化图像

```python
def classify_image(image: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """分类图像（Gradio 接口函数）"""
    if classifier is None:
        return {"错误": 1.0}, np.zeros((224, 224, 3), dtype=np.uint8)
    
    try:
        predicted_class, confidences, visualization = classifier.predict(image)
        return confidences, visualization
    except Exception as e:
        logger.error(f"分类时出错: {e}")
        return {"错误": 1.0}, image
```

#### 2.3 界面创建函数

**create_gradio_interface() 函数** (第201-255行):

这是创建Gradio界面的核心函数，定义了界面的所有元素：

```python
interface = gr.Interface(
    fn=classify_image,                          # 处理函数
    inputs=gr.Image(type="numpy", label="上传皮肤病变图像"),  # 输入组件
    outputs=[
        gr.Label(num_top_classes=5, label="分类结果与置信度"),  # 输出1：分类结果
        gr.Image(type="numpy", label="Grad-CAM 可视化")        # 输出2：热力图
    ],
    title=title,              # 标题
    description=description,  # 使用说明
    article=article,          # 技术说明
    examples=[],              # 示例图像
    allow_flagging="never",   # 禁用标记功能
    analytics_enabled=False   # 禁用分析
)
```

**界面包含**:
- **标题**: "🔬 皮肤病变智能分类系统"
- **描述**: 使用方法和注意事项（第206-220行）
- **输入**: 图像上传组件
- **输出**: 
  - 分类结果标签（显示前5个类别的置信度）
  - Grad-CAM可视化图像
- **技术说明**: 模型架构、性能指标等（第222-235行）

#### 2.4 主函数入口

**main() 函数** (第258-288行):

应用程序的启动入口，处理命令行参数并启动服务器：

```python
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='皮肤病变分类 Web 应用')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--model_type', type=str, default='resnet50')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    
    args = parser.parse_args()
    
    # 初始化分类器
    initialize_classifier(
        model_path=args.model_path,
        model_type=args.model_type
    )
    
    # 创建并启动界面
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",  # 监听所有网络接口
        server_port=args.port,   # 端口号
        share=args.share         # 是否创建公共链接
    )
```

## 工作流程

### 完整的数据流程：

```
1. 用户上传图像
   ↓
2. Gradio接收图像 (numpy数组格式)
   ↓
3. 调用 classify_image(image)
   ↓
4. 调用 classifier.predict(image)
   ↓
5. 图像预处理:
   - 调整大小到224x224
   - 标准化 (ImageNet均值/标准差)
   - 转换为PyTorch张量
   ↓
6. 模型推理:
   - 前向传播
   - Softmax计算概率
   ↓
7. 生成Grad-CAM:
   - 反向传播获取梯度
   - 计算加权激活图
   - 生成热力图
   - 叠加到原图
   ↓
8. 返回结果:
   - 置信度字典 {"良性": 0.85, "恶性": 0.15}
   - Grad-CAM可视化图像
   ↓
9. Gradio显示:
   - 左侧：分类标签和置信度
   - 右侧：热力图可视化
```

## 启动方法

### 方法1：使用命令行参数启动

```bash
# 基本启动（使用默认参数）
python web_app/app.py

# 指定模型路径和类型
python web_app/app.py --model_path checkpoints/best_model.pth --model_type resnet50

# 指定端口
python web_app/app.py --port 7860

# 创建公共分享链接
python web_app/app.py --share
```

### 方法2：使用启动脚本

创建 `launch_web_app.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_app.app import initialize_classifier, create_gradio_interface

if __name__ == "__main__":
    print("="*60)
    print("🔬 启动皮肤病变智能分类系统")
    print("="*60)
    
    # 初始化分类器（使用演示模式，不需要真实模型）
    initialize_classifier(
        model_path='checkpoints/best_model.pth',
        model_type='resnet50'
    )
    
    # 创建并启动界面
    interface = create_gradio_interface()
    
    print("\n✅ 界面已准备就绪！")
    print("📍 本地访问: http://localhost:7860")
    print("🌐 网络访问: http://0.0.0.0:7860")
    print("\n按 Ctrl+C 停止服务器\n")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

然后运行:
```bash
python launch_web_app.py
```

### 方法3：在Jupyter Notebook中使用

```python
import sys
sys.path.append('.')

from web_app.app import initialize_classifier, create_gradio_interface

# 初始化
initialize_classifier(
    model_path='checkpoints/best_model.pth',
    model_type='resnet50'
)

# 创建界面（在notebook中自动内联显示）
interface = create_gradio_interface()
interface.launch(inline=True)
```

## 界面元素详解

### 输入区域

**图像上传组件** (`gr.Image`):
- 类型: numpy数组
- 支持格式: JPG, PNG, JPEG
- 标签: "上传皮肤病变图像"
- 用户可以：
  - 拖拽上传
  - 点击浏览
  - 使用摄像头拍照（如果支持）

### 输出区域

**输出1 - 分类结果** (`gr.Label`):
- 显示前5个类别（如果有）
- 格式: 
  ```
  良性 (Benign): 85.2%
  恶性 (Malignant): 14.8%
  ```
- 自动排序：置信度从高到低

**输出2 - Grad-CAM可视化** (`gr.Image`):
- 显示热力图叠加的图像
- 红色区域：模型高度关注的区域
- 蓝色区域：模型较少关注的区域
- 帮助理解模型决策依据

### 界面布局

```
┌─────────────────────────────────────────────────────────────┐
│     🔬 皮肤病变智能分类系统                                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ## 智能皮肤癌检测与筛查系统                                  │
│  使用方法、注意事项...                                        │
│                                                               │
├───────────────────────┬─────────────────────────────────────┤
│                       │                                       │
│   [上传皮肤病变图像]   │   [分类结果与置信度]                 │
│                       │   良性: 85%                          │
│   ┌─────────────┐    │   恶性: 15%                          │
│   │             │    │                                       │
│   │   图像区域   │    │   [Grad-CAM 可视化]                  │
│   │             │    │   ┌─────────────┐                    │
│   │             │    │   │   热力图    │                    │
│   └─────────────┘    │   │   叠加图像  │                    │
│                       │   └─────────────┘                    │
│   [Submit] [Clear]    │                                       │
│                       │                                       │
├───────────────────────┴─────────────────────────────────────┤
│  ### 技术说明                                                 │
│  模型架构、性能指标...                                        │
└─────────────────────────────────────────────────────────────┘
```

## 技术依赖

### 核心依赖
- `gradio`: Web界面框架
- `torch`: 深度学习框架
- `numpy`: 数值计算
- `opencv-python`: 图像处理
- `PIL`: 图像加载

### 模块依赖
- `models.cnn_model`: 模型定义
- `data_processing.preprocess`: 数据预处理
- `visualization.grad_cam`: Grad-CAM可视化

## 自定义界面

### 修改类别名称

编辑 `initialize_classifier()` 函数（第161-177行）：

```python
def initialize_classifier(
    model_path: str = 'checkpoints/best_model.pth',
    model_type: str = 'resnet50',
    class_names: list = None
):
    global classifier
    
    if class_names is None:
        # 自定义类别名称
        class_names = ['良性 (Benign)', '恶性 (Malignant)']
    
    classifier = SkinLesionClassifier(
        model_path=model_path,
        model_type=model_type,
        num_classes=len(class_names),
        class_names=class_names
    )
```

### 修改界面文本

编辑 `create_gradio_interface()` 函数中的文本（第204-235行）：

```python
# 修改标题
title = "🔬 皮肤病变智能分类系统"

# 修改描述
description = """
你的自定义描述...
"""

# 修改技术说明
article = """
你的自定义技术说明...
"""
```

### 添加示例图像

在 `create_gradio_interface()` 函数中（第248-250行）：

```python
examples=[
    ["examples/benign_example1.jpg"],
    ["examples/malignant_example1.jpg"],
]
```

### 修改默认端口

在 `main()` 函数中（第267行）：

```python
parser.add_argument('--port', type=int, default=8080,  # 改为8080
                    help='端口号')
```

## 常见问题

### Q1: 如何在没有训练模型的情况下测试界面？

A: 修改 `initialize_classifier()` 使其在模型不存在时仍能工作：

```python
# 第64-65行
else:
    logger.warning(f"模型文件不存在: {model_path}，使用未训练的模型")
```

程序会使用未训练的模型（随机权重），虽然预测结果无意义，但界面可以正常显示。

### Q2: 如何加快推理速度？

A: 
1. 使用GPU（自动检测）
2. 使用更小的模型（如resnet18而不是resnet50）
3. 减小输入图像大小

### Q3: 如何部署到服务器？

A: 使用 `--share` 参数创建公共链接：

```bash
python web_app/app.py --share
```

Gradio会生成一个临时的公共URL（有效期72小时）。

### Q4: 如何添加用户认证？

A: 在 `interface.launch()` 中添加认证：

```python
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    auth=("username", "password")  # 添加认证
)
```

## 性能优化

### 1. 批处理支持

修改 `classify_image()` 支持批量处理：

```python
def classify_image_batch(images: list) -> list:
    results = []
    for image in images:
        confidences, visualization = classify_image(image)
        results.append((confidences, visualization))
    return results
```

### 2. 缓存机制

添加结果缓存避免重复计算：

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def classify_image_cached(image_hash):
    # 实现缓存逻辑
    pass
```

### 3. 异步处理

使用Gradio的异步功能：

```python
import asyncio

async def classify_image_async(image):
    # 异步实现
    pass

interface = gr.Interface(
    fn=classify_image_async,
    # ...
)
```

## 总结

Web应用的核心功能都在 `web_app/app.py` 文件中实现：

- **SkinLesionClassifier类**: 封装模型加载、预测和Grad-CAM生成
- **classify_image()函数**: Gradio接口函数，连接界面和分类器
- **create_gradio_interface()函数**: 创建界面布局和组件
- **main()函数**: 程序入口，处理启动参数

启动后，访问 http://localhost:7860 即可使用完整的Web界面进行皮肤病变分类和可视化。
