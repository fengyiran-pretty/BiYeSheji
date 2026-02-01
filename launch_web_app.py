#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
皮肤病变分类系统 - Web应用启动脚本
Launch script for Skin Lesion Classification Web Application

使用方法:
    python launch_web_app.py
    python launch_web_app.py --model_type resnet50
    python launch_web_app.py --port 8080
    python launch_web_app.py --demo  # 演示模式（不需要真实模型）
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_app.app import initialize_classifier, create_gradio_interface
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      🔬 皮肤病变智能分类系统 Web 应用                         ║
║      Skin Lesion Classification System                      ║
║                                                              ║
║      基于深度学习的皮肤癌早期检测与筛查                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_info(args):
    """打印配置信息"""
    print("\n📋 配置信息:")
    print(f"   模型类型: {args.model_type}")
    print(f"   模型路径: {args.model_path}")
    print(f"   端口号: {args.port}")
    print(f"   共享链接: {'是' if args.share else '否'}")
    print(f"   演示模式: {'是' if args.demo else '否'}")
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='皮肤病变分类 Web 应用启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  基本启动:
    python launch_web_app.py
  
  指定模型:
    python launch_web_app.py --model_type efficientnet_b0
  
  指定端口:
    python launch_web_app.py --port 8080
  
  演示模式（不需要训练好的模型）:
    python launch_web_app.py --demo
  
  创建公共分享链接:
    python launch_web_app.py --share
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='checkpoints/best_model.pth',
        help='模型权重文件路径 (默认: checkpoints/best_model.pth)'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='resnet50',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                 'efficientnet_b3', 'efficientnet_b4',
                 'densenet121', 'densenet169', 'densenet201'],
        help='模型架构类型 (默认: resnet50)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Web服务器端口号 (默认: 7860)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='创建公共分享链接（Gradio临时链接，有效期72小时）'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='演示模式：即使没有训练好的模型也可以运行（用于界面演示）'
    )
    
    parser.add_argument(
        '--class_names',
        type=str,
        nargs='+',
        default=None,
        help='类别名称列表 (默认: ["良性 (Benign)", "恶性 (Malignant)"])'
    )
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 打印配置信息
    print_info(args)
    
    try:
        # 检查模型文件
        if not args.demo and not os.path.exists(args.model_path):
            logger.warning(f"⚠️  警告: 模型文件不存在: {args.model_path}")
            logger.warning("   程序将使用未训练的模型（预测结果无意义）")
            logger.warning("   如果只想查看界面，可以使用 --demo 参数")
            print()
            response = input("是否继续？(y/n): ")
            if response.lower() != 'y':
                print("已取消启动")
                return
            print()
        
        # 初始化分类器
        logger.info("🔧 正在初始化分类器...")
        initialize_classifier(
            model_path=args.model_path,
            model_type=args.model_type,
            class_names=args.class_names
        )
        logger.info("✅ 分类器初始化完成")
        
        # 创建界面
        logger.info("🎨 正在创建Web界面...")
        interface = create_gradio_interface()
        logger.info("✅ 界面创建完成")
        
        # 打印访问信息
        print("\n" + "="*60)
        print("🚀 Web应用已启动!")
        print("="*60)
        print(f"\n📍 本地访问地址:")
        print(f"   http://localhost:{args.port}")
        print(f"   http://127.0.0.1:{args.port}")
        
        if args.share:
            print(f"\n🌐 公共分享链接:")
            print(f"   （启动后将显示在下方）")
        
        print("\n💡 使用提示:")
        print("   1. 在浏览器中打开上述地址")
        print("   2. 上传皮肤病变图像")
        print("   3. 查看分类结果和Grad-CAM可视化")
        print("\n⏹️  按 Ctrl+C 停止服务器")
        print("="*60 + "\n")
        
        # 启动界面
        interface.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 收到停止信号，正在关闭服务器...")
        print("✅ 服务器已停止")
        
    except Exception as e:
        logger.error(f"\n❌ 启动失败: {e}")
        logger.error(f"   错误类型: {type(e).__name__}")
        import traceback
        logger.error(f"   详细信息:\n{traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
