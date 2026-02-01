"""
Web 应用包初始化
"""

from .app import (
    SkinLesionClassifier,
    initialize_classifier,
    classify_image,
    create_gradio_interface
)

__all__ = [
    'SkinLesionClassifier',
    'initialize_classifier',
    'classify_image',
    'create_gradio_interface'
]
