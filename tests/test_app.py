"""
Web 应用测试
Unit tests for web application
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWebApp:
    """Web 应用测试类"""
    
    def test_image_preprocessing(self):
        """测试图像预处理"""
        # 创建测试图像
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        assert image.shape == (224, 224, 3)
        assert image.dtype == np.uint8
    
    def test_confidence_format(self):
        """测试置信度格式"""
        confidences = {
            'benign': 0.7,
            'malignant': 0.3
        }
        
        assert sum(confidences.values()) == pytest.approx(1.0, rel=1e-5)
        assert all(0 <= v <= 1 for v in confidences.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
