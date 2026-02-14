# -*- coding: utf-8 -*-
"""
Retinex图像增强算法 - 主程序
========================================
该程序提供图形用户界面，用于对图像进行Retinex增强处理。

依赖库:
    - OpenCV (cv2): 图像读取、处理和保存
    - NumPy: 数值计算
    - PyQt5: 图形用户界面

使用方法:
    1. 运行 python run.py
    2. 点击"选择图片"上传图像
    3. 选择增强算法
    4. 调整参数（可选）
    5. 点击"处理图像"查看结果
    6. 点击"保存结果"下载处理后的图像
"""



import retinex
from ui import RetinexGUI, get_app_style
from PyQt5.QtWidgets import QApplication
import sys

def main():
    """
    主函数 - 创建并运行GUI应用程序
    """
    # 创建Qt应用
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置应用样式
    app.setStyleSheet(get_app_style())

    # 创建并显示主窗口
    window = RetinexGUI(retinex)
    window.show()

    # 运行事件循环
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
