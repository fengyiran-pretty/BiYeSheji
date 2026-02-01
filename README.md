# 皮肤病变智能分类系统 - 毕业设计

基于深度学习的皮肤癌（黑色素瘤）早期检测与筛查系统

## 📋 项目简介

本项目是一个智能系统，利用深度学习技术自动分析皮肤病变图像，以辅助皮肤癌（特别是黑色素瘤）的早期检测与筛查。系统使用卷积神经网络（CNN）对皮肤病变图像进行良性/恶性分类，并提供用户友好的 Web 界面。

### 核心功能

- ✅ **深度学习模型**: 基于 ResNet/EfficientNet 的 CNN 分类模型
- ✅ **数据增强**: 使用 Albumentations 进行高级图像增强
- ✅ **模型训练**: 完整的训练、验证和早停机制
- ✅ **性能评估**: 准确率、精确率、召回率、F1-Score、AUC 等指标
- ✅ **可视化**: Grad-CAM 热力图展示模型决策依据
- ✅ **Web 界面**: Gradio 实现的用户友好界面
- ✅ **GPU 加速**: 支持 CUDA 加速训练和推理

## 🏗️ 项目结构

```
BiYeSheji/
├── README.md                 # 项目文档
├── requirements.txt          # 依赖列表
├── data_processing/          # 数据预处理模块
│   ├── preprocess.py         # 数据清洗和增强
│   └── split_dataset.py      # 数据集划分
├── models/                   # 模型实现
│   ├── cnn_model.py          # CNN 模型定义
│   ├── train.py              # 训练脚本
│   └── evaluate.py           # 评估脚本
├── visualization/            # 可视化工具
│   └── grad_cam.py           # Grad-CAM 热力图
├── web_app/                  # Web 应用
│   ├── app.py                # Gradio 应用
│   ├── templates/            # HTML 模板
│   └── static/               # 静态资源
└── tests/                    # 测试脚本
    ├── test_models.py        # 模型测试
    └── test_app.py           # 应用测试
```

## 🚀 快速开始

### 在WSL中安装（推荐Windows用户）

如果您使用Windows系统，建议在WSL（Windows Subsystem for Linux）中运行此项目以获得最佳性能。

**🎯 一键安装**:
```bash
# 在WSL终端中运行
bash <(curl -s https://raw.githubusercontent.com/fengyiran-pretty/BiYeSheji/main/wsl_setup.sh)
```

**📚 详细指南**:
- [WSL项目复制指南](WSL项目复制指南.md) - 完整的安装和配置文档
- [WSL快速参考](WSL快速参考.txt) - 一页纸速查卡片
- [WSL图解教程](WSL图解教程.md) - 图文并茂的详细教程

**手动安装**:
```bash
# 1. 克隆项目
cd ~/projects
git clone https://github.com/fengyiran-pretty/BiYeSheji.git
cd BiYeSheji

# 2. 安装依赖
pip3 install -r requirements.txt

# 3. 运行
python3 persistent_server.py
```

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于 GPU 加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

1. 下载 ISIC 数据集：[https://www.isic-archive.com/](https://www.isic-archive.com/)
2. 组织数据集结构：

```
data/
├── raw/
│   ├── benign/
│   │   ├── image1.jpg
│   │   └── ...
│   └── malignant/
│       ├── image2.jpg
│       └── ...
```

3. 划分数据集：

```bash
python -c "
from data_processing.split_dataset import split_dataset
split_dataset('data/raw', 'data/processed', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
"
```

### 模型训练

创建训练脚本 `train_model.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import get_model
from models.train import Trainer, create_data_loaders
from data_processing.preprocess import get_training_augmentation, get_validation_augmentation

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型
model = get_model('resnet50', num_classes=2, pretrained=True)
model = model.to(device)

# 数据加载器
train_transform = get_training_augmentation()
val_transform = get_validation_augmentation()
train_loader, val_loader = create_data_loaders(
    'data/processed/train',
    'data/processed/validation',
    train_transform,
    val_transform,
    batch_size=32
)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

# 训练
trainer = Trainer(
    model=model,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=50,
    save_dir='checkpoints'
)

history = trainer.train()
```

运行训练：

```bash
python train_model.py
```

### 模型评估

```python
from models.cnn_model import get_model
from models.evaluate import ModelEvaluator, load_checkpoint
from models.train import create_data_loaders
from data_processing.preprocess import get_validation_augmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = get_model('resnet50', num_classes=2)
model = load_checkpoint('checkpoints/best_model.pth', model, device)

# 创建测试数据加载器
test_transform = get_validation_augmentation()
_, test_loader = create_data_loaders(
    'data/processed/validation',  # 占位
    'data/processed/test',
    test_transform,
    test_transform,
    batch_size=32
)

# 评估
evaluator = ModelEvaluator(model, device, ['良性', '恶性'])
results = evaluator.evaluate(test_loader)
evaluator.save_results(results, 'results')
```

### 启动 Web 应用

```bash
# 方法1: 直接运行
python web_app/app.py --model_path checkpoints/best_model.pth --model_type resnet50 --port 7860

# 方法2: 使用启动脚本（推荐）
python launch_web_app.py

# 方法3: 演示模式（不需要模型）
python demo_web_app.py
```

访问 `http://localhost:7860` 使用 Web 界面。

**详细文档**:
- [界面功能实现总结](界面功能实现总结.md) - Web应用完整说明
- [快速参考](快速参考.md) - 一页纸速查手册
- [WEB应用使用说明](WEB应用使用说明.md) - 详细使用指南
- [界面展示图解](界面展示图解.md) - 界面结构和代码图解

## 📊 性能指标

预期性能指标（在 ISIC 数据集上）：

| 指标 | 目标值 |
|------|--------|
| 准确率 (Accuracy) | > 85% |
| AUC | > 0.85 |
| 精确率 (Precision) | > 80% |
| 召回率 (Recall) | > 80% |
| F1-Score | > 80% |

## 🔬 技术栈

- **深度学习框架**: PyTorch 2.0+
- **模型库**: timm (PyTorch Image Models)
- **图像增强**: Albumentations
- **Web 框架**: Gradio
- **可视化**: Matplotlib, Seaborn
- **科学计算**: NumPy, Pandas
- **评估指标**: scikit-learn

## 📖 使用说明

### 模型选择

支持多种预训练模型：

- `resnet18`, `resnet34`, `resnet50`, `resnet101`
- `efficientnet_b0` 至 `efficientnet_b7`
- `densenet121`, `densenet169`, `densenet201`
- `basic_cnn` (基础 CNN，用于概念验证)

### 数据增强

训练时使用的数据增强包括：

- 随机水平/垂直翻转
- 随机旋转（90度）
- 随机缩放和平移
- 噪声和模糊
- 光学畸变
- 颜色抖动

### Grad-CAM 可视化

使用 Grad-CAM 技术生成热力图，展示模型关注的区域：

```python
from visualization.grad_cam import visualize_gradcam, get_target_layer

target_layer = get_target_layer(model, 'resnet50')
visualize_gradcam(
    model,
    image_tensor,
    original_image,
    ['良性', '恶性'],
    target_layer,
    save_path='gradcam_result.png'
)
```

## 🧪 测试

运行测试：

```bash
pytest tests/ -v
```

## 📝 注意事项

1. **医疗免责声明**: 本系统仅供辅助参考，不能替代专业医生诊断
2. **数据隐私**: 请遵守数据使用协议和隐私法规
3. **模型性能**: 实际性能取决于训练数据质量和数量
4. **GPU 内存**: 大模型（如 EfficientNet-B7）需要更多 GPU 内存

## 🔗 参考资源

- [ISIC 数据集](https://www.isic-archive.com/)
- [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)
- [MONAI 框架](https://github.com/Project-MONAI/MONAI)
- [ResNet 论文](https://arxiv.org/abs/1512.03385)
- [EfficientNet 论文](https://arxiv.org/abs/1905.11946)
- [Grad-CAM 论文](https://arxiv.org/abs/1610.02391)

## 👨‍💻 开发者

封怡然 - 毕业设计项目

## 📄 许可证

本项目遵循开源协议。部分代码参考了开源项目，请查看相应的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**开发时间**: 2024-2026  
**框架**: PyTorch + Gradio  
**目标**: 辅助皮肤癌早期检测
