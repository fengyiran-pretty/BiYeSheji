"""
模型评估模块
Model evaluation module for skin lesion classification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str]
    ):
        """
        初始化评估器
        
        Args:
            model: PyTorch 模型
            device: 评估设备
            class_names: 类别名称列表
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对数据集进行预测
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            真实标签、预测标签、预测概率
        """
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return (
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """
        评估模型性能
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            包含各项指标的字典
        """
        logger.info("开始评估模型...")
        
        # 获取预测结果
        y_true, y_pred, y_prob = self.predict(data_loader)
        
        # 计算各项指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 计算每个类别的指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # 计算 AUC (如果是二分类问题)
        auc_score = None
        if len(self.class_names) == 2:
            auc_score = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # 多分类问题使用 one-vs-rest
            try:
                auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except:
                logger.warning("无法计算多分类 AUC")
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 构建结果字典
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc_score) if auc_score is not None else None,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i])
                }
                for i in range(len(self.class_names))
            }
        }
        
        # 打印结果
        logger.info("\n" + "="*50)
        logger.info("评估结果:")
        logger.info("="*50)
        logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
        logger.info(f"精确率 (Precision): {precision:.4f}")
        logger.info(f"召回率 (Recall): {recall:.4f}")
        logger.info(f"F1 分数: {f1:.4f}")
        if auc_score is not None:
            logger.info(f"AUC: {auc_score:.4f}")
        logger.info("\n每个类别的指标:")
        for class_name in self.class_names:
            metrics = results['per_class_metrics'][class_name]
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    F1-Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = False
    ):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            save_path: 保存路径
            normalize: 是否归一化
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': '数量' if not normalize else '比例'}
        )
        plt.title('混淆矩阵' + (' (归一化)' if normalize else ''))
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        绘制 ROC 曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        if len(self.class_names) == 2:
            # 二分类
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc = roc_auc_score(y_true, y_prob[:, 1])
            plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        else:
            # 多分类
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            
            for i in range(len(self.class_names)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)')
        plt.ylabel('真阳性率 (True Positive Rate)')
        plt.title('ROC 曲线')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC 曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
            
        Returns:
            分类报告字符串
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        logger.info("\n分类报告:")
        logger.info("\n" + report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"分类报告已保存到: {save_path}")
        
        return report
    
    def save_results(self, results: Dict, output_dir: str):
        """
        保存评估结果
        
        Args:
            results: 评估结果字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 JSON 格式的结果
        json_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"评估结果已保存到: {json_path}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device) -> nn.Module:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: PyTorch 模型
        device: 设备
        
    Returns:
        加载权重后的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"已加载检查点: {checkpoint_path}")
    if 'epoch' in checkpoint:
        logger.info(f"  Epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        logger.info(f"  验证准确率: {checkpoint['val_acc']:.4f}")
    
    return model


if __name__ == "__main__":
    print("模型评估模块已加载")
    print("使用示例:")
    print("  from models.cnn_model import get_model")
    print("  from models.evaluate import ModelEvaluator, load_checkpoint")
    print("  ")
    print("  model = get_model('resnet50', num_classes=2)")
    print("  model = load_checkpoint('checkpoints/best_model.pth', model, device)")
    print("  evaluator = ModelEvaluator(model, device, ['benign', 'malignant'])")
    print("  results = evaluator.evaluate(test_loader)")
    print("  evaluator.save_results(results, 'results')")
