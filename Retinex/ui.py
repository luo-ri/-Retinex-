# -*- coding: utf-8 -*-
"""
Retinex图像增强算法 - GUI界面模块
========================================
该模块提供图形用户界面，支持以下功能：
1. 上传图片进行Retinex增强处理
2. 选择三种不同的算法
3. 实时展示原图和处理结果对比
4. 自定义算法参数
5. 保存处理后的图像

依赖库:
    - PyQt5: 图形用户界面
"""

import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QSlider, QGroupBox, QFileDialog, QMessageBox,
                             QGridLayout, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import json


class RetinexGUI(QMainWindow):
    """
    Retinex图像增强GUI主窗口类
    """

    def __init__(self, retinex_module):
        """
        初始化GUI窗口

        参数:
            retinex_module: retinex算法模块引用
        """
        super().__init__()
        self.retinex = retinex_module
        self.current_img = None
        self.result_img = None
        self.config = self.load_config()
        self.init_ui()

    def load_config(self):
        """
        从配置文件加载默认参数

        返回:
            配置参数字典
        """
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except:
            return {
                "sigma_list": [15, 80, 250],
                "G": 5.0,
                "b": 25.0,
                "alpha": 125.0,
                "beta": 46.0,
                "low_clip": 0.01,
                "high_clip": 0.99
            }

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('Retinex图像增强工具')
        self.setGeometry(100, 100, 1400, 800)

        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, 1)

        # 右侧图像显示区域
        image_panel = self.create_image_panel()
        layout.addWidget(image_panel, 3)

    def create_control_panel(self):
        """
        创建左侧控制面板

        返回:
            控制面板widget
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 图片上传区域
        upload_group = QGroupBox('图片上传')
        upload_layout = QVBoxLayout()
        self.upload_btn = QPushButton('选择图片')
        self.upload_btn.clicked.connect(self.select_image)
        self.upload_btn.setMinimumHeight(50)
        upload_layout.addWidget(self.upload_btn)
        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)

        # 算法选择区域
        algo_group = QGroupBox('算法选择')
        algo_layout = QVBoxLayout()
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            'MSRCR - 带颜色恢复的多尺度Retinex',
            'Automated MSRCR - 自动化多尺度Retinex',
            'MSRCP - 带颜色保持的多尺度Retinex'
        ])
        self.algo_combo.currentIndexChanged.connect(self.on_algo_changed)
        algo_layout.addWidget(self.algo_combo)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # 参数设置区域
        param_group = self.create_param_panel()
        layout.addWidget(param_group)

        # 操作按钮
        action_group = QGroupBox('操作')
        action_layout = QVBoxLayout()

        self.process_btn = QPushButton('处理图像')
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setEnabled(False)
        action_layout.addWidget(self.process_btn)

        self.save_btn = QPushButton('保存结果')
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setMinimumHeight(50)
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.save_btn)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        layout.addStretch()
        return panel

    def create_param_panel(self):
        """
        创建参数设置面板

        返回:
            参数面板widget
        """
        panel = QGroupBox('参数设置')
        layout = QVBoxLayout()

        # 高斯标准差列表
        layout.addWidget(QLabel('高斯标准差 Sigma (逗号分隔):'))
        self.sigma_label = QLabel(f"{self.config['sigma_list'][0]}, {self.config['sigma_list'][1]}, {self.config['sigma_list'][2]}")
        self.sigma_label.setStyleSheet('color: #666; font-size: 11px;')
        layout.addWidget(self.sigma_label)

        # G 增益系数
        layout.addWidget(QLabel('增益系数 G:'))
        self.g_slider = QSlider(Qt.Horizontal)
        self.g_slider.setRange(1, 200)
        self.g_slider.setValue(int(self.config['G'] * 10))
        self.g_label = QLabel(f'{self.config["G"]:.1f}')
        g_layout = QHBoxLayout()
        g_layout.addWidget(self.g_slider)
        g_layout.addWidget(self.g_label)
        layout.addLayout(g_layout)
        self.g_slider.valueChanged.connect(lambda v: self.update_param_label(self.g_label, v / 10))

        # b 偏差系数
        layout.addWidget(QLabel('偏差系数 b:'))
        self.b_slider = QSlider(Qt.Horizontal)
        self.b_slider.setRange(-100, 100)
        self.b_slider.setValue(int(self.config['b']))
        self.b_label = QLabel(f'{self.config["b"]:.1f}')
        b_layout = QHBoxLayout()
        b_layout.addWidget(self.b_slider)
        b_layout.addWidget(self.b_label)
        layout.addLayout(b_layout)
        self.b_slider.valueChanged.connect(lambda v: self.update_param_label(self.b_label, v))

        # alpha 颜色强度调整系数
        layout.addWidget(QLabel('颜色强度 alpha:'))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(10, 250)
        self.alpha_slider.setValue(int(self.config['alpha']))
        self.alpha_label = QLabel(f'{self.config["alpha"]:.1f}')
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addWidget(self.alpha_label)
        layout.addLayout(alpha_layout)
        self.alpha_slider.valueChanged.connect(lambda v: self.update_param_label(self.alpha_label, v))

        # beta 颜色平衡系数
        layout.addWidget(QLabel('颜色平衡 beta:'))
        self.beta_slider = QSlider(Qt.Horizontal)
        self.beta_slider.setRange(10, 100)
        self.beta_slider.setValue(int(self.config['beta']))
        self.beta_label = QLabel(f'{self.config["beta"]:.1f}')
        beta_layout = QHBoxLayout()
        beta_layout.addWidget(self.beta_slider)
        beta_layout.addWidget(self.beta_label)
        layout.addLayout(beta_layout)
        self.beta_slider.valueChanged.connect(lambda v: self.update_param_label(self.beta_label, v))

        # low_clip 低裁剪值
        layout.addWidget(QLabel('低裁剪值 low_clip:'))
        self.low_clip_slider = QSlider(Qt.Horizontal)
        self.low_clip_slider.setRange(1, 50)
        self.low_clip_slider.setValue(int(self.config['low_clip'] * 100))
        self.low_clip_label = QLabel(f'{self.config["low_clip"]:.2f}')
        low_clip_layout = QHBoxLayout()
        low_clip_layout.addWidget(self.low_clip_slider)
        low_clip_layout.addWidget(self.low_clip_label)
        layout.addLayout(low_clip_layout)
        self.low_clip_slider.valueChanged.connect(lambda v: self.update_param_label(self.low_clip_label, v / 100))

        # high_clip 高裁剪值
        layout.addWidget(QLabel('高裁剪值 high_clip:'))
        self.high_clip_slider = QSlider(Qt.Horizontal)
        self.high_clip_slider.setRange(50, 99)
        self.high_clip_slider.setValue(int(self.config['high_clip'] * 100))
        self.high_clip_label = QLabel(f'{self.config["high_clip"]:.2f}')
        high_clip_layout = QHBoxLayout()
        high_clip_layout.addWidget(self.high_clip_slider)
        high_clip_layout.addWidget(self.high_clip_label)
        layout.addLayout(high_clip_layout)
        self.high_clip_slider.valueChanged.connect(lambda v: self.update_param_label(self.high_clip_label, v / 100))

        panel.setLayout(layout)
        self.param_panel = panel
        return panel

    def create_image_panel(self):
        """
        创建右侧图像显示面板

        返回:
            图像面板widget
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 原始图像
        orig_layout = QVBoxLayout()
        orig_label = QLabel('原始图像')
        orig_label.setStyleSheet('font-weight: bold; font-size: 14px;')
        orig_layout.addWidget(orig_label)

        self.orig_label = QLabel('请选择图片')
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setMinimumSize(400, 300)
        self.orig_label.setStyleSheet('''
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                color: #999;
                font-size: 14px;
            }
        ''')
        orig_layout.addWidget(self.orig_label)
        layout.addLayout(orig_layout)

        # 结果图像
        result_layout = QVBoxLayout()
        result_label = QLabel('处理结果')
        result_label.setStyleSheet('font-weight: bold; font-size: 14px;')
        result_layout.addWidget(result_label)

        self.result_display = QLabel('等待处理')
        self.result_display.setAlignment(Qt.AlignCenter)
        self.result_display.setMinimumSize(400, 300)
        self.result_display.setStyleSheet('''
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                color: #999;
                font-size: 14px;
            }
        ''')
        result_layout.addWidget(self.result_display)
        layout.addLayout(result_layout)

        return panel

    def update_param_label(self, label, value):
        """
        更新参数标签显示

        参数:
            label: 标签控件
            value: 参数值
        """
        label.setText(f'{value:.2f}')

    def select_image(self):
        """选择图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择图片',
            '',
            '图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)'
        )

        if file_path:
            try:
                # 读取图片
                self.current_img = cv2.imread(file_path)
                if self.current_img is None:
                    QMessageBox.warning(self, '错误', '无法读取图片文件')
                    return

                # 显示原始图片
                self.display_image(self.current_img, self.orig_label)

                # 清空结果显示
                self.result_display.clear()
                self.result_display.setText('等待处理')
                self.result_display.setStyleSheet('''
                    QLabel {
                        background-color: #f0f0f0;
                        border: 2px dashed #ccc;
                        color: #999;
                        font-size: 14px;
                    }
                ''')
                self.result_img = None
                self.save_btn.setEnabled(False)
                self.process_btn.setEnabled(True)

                print(f'已加载图片: {file_path}')
                print(f'图片尺寸: {self.current_img.shape}')

            except Exception as e:
                QMessageBox.critical(self, '错误', f'读取图片失败: {str(e)}')

    def display_image(self, img, label):
        """
        在标签中显示图像

        参数:
            img: OpenCV图像数组
            label: 要显示的标签控件
        """
        # 转换为RGB格式（OpenCV使用BGR）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 缩放图像以适应标签大小
        label_size = label.size()
        h, w = img_rgb.shape[:2]

        # 保持宽高比缩放
        scale = min(label_size.width() / w, label_size.height() / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 转换为QPixmap
        h, w, c = img_scaled.shape
        bytes_per_line = c * w
        q_img = QImage(img_scaled.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 设置样式
        label.setStyleSheet('''
            QLabel {
                background-color: #fff;
                border: 1px solid #ccc;
            }
        ''')

        # 显示图像
        label.setPixmap(QPixmap.fromImage(q_img))

    def on_algo_changed(self, index):
        """
        算法选择变化时的处理

        参数:
            index: 选中的算法索引
        """
        # 根据选择的算法启用/禁用相应参数
        # MSRCR: 所有参数可用
        # Automated MSRCR: 禁用G, b, alpha, beta
        # MSRCP: 禁用G, b, alpha, beta
        if index == 0:  # MSRCR
            self.enable_all_params()
        elif index == 1:  # Automated MSRCR
            self.disable_algo_params()
        elif index == 2:  # MSRCP
            self.disable_algo_params()

    def enable_all_params(self):
        """启用所有参数控件"""
        self.g_slider.setEnabled(True)
        self.b_slider.setEnabled(True)
        self.alpha_slider.setEnabled(True)
        self.beta_slider.setEnabled(True)
        self.low_clip_slider.setEnabled(True)
        self.high_clip_slider.setEnabled(True)

    def disable_algo_params(self):
        """禁用算法特定参数（G, b, alpha, beta）"""
        self.g_slider.setEnabled(False)
        self.b_slider.setEnabled(False)
        self.alpha_slider.setEnabled(False)
        self.beta_slider.setEnabled(False)
        self.low_clip_slider.setEnabled(True)
        self.high_clip_slider.setEnabled(True)

    def process_image(self):
        """处理图像"""
        if self.current_img is None:
            QMessageBox.warning(self, '警告', '请先选择图片')
            return

        try:
            # 获取当前参数
            sigma_list = self.config['sigma_list']
            G = self.g_slider.value() / 10
            b = float(self.b_slider.value())
            alpha = float(self.alpha_slider.value())
            beta = float(self.beta_slider.value())
            low_clip = self.low_clip_slider.value() / 100
            high_clip = self.high_clip_slider.value() / 100

            # 获取选择的算法
            algo_index = self.algo_combo.currentIndex()

            print(f'开始处理图像，使用算法: {self.algo_combo.currentText()}')

            # 根据选择的算法执行处理
            if algo_index == 0:  # MSRCR
                self.result_img = self.retinex.MSRCR(
                    self.current_img,
                    sigma_list,
                    G, b, alpha, beta, low_clip, high_clip
                )
            elif algo_index == 1:  # Automated MSRCR
                self.result_img = self.retinex.automatedMSRCR(
                    self.current_img,
                    sigma_list
                )
            elif algo_index == 2:  # MSRCP
                self.result_img = self.retinex.MSRCP(
                    self.current_img,
                    sigma_list,
                    low_clip, high_clip
                )

            # 显示处理结果
            self.display_image(self.result_img, self.result_display)
            self.save_btn.setEnabled(True)

            print('图像处理完成')

        except Exception as e:
            QMessageBox.critical(self, '错误', f'处理图像失败: {str(e)}')
            print(f'错误: {str(e)}')

    def save_result(self):
        """保存处理结果"""
        if self.result_img is None:
            QMessageBox.warning(self, '警告', '没有可保存的结果')
            return

        try:
            # 获取保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                '保存处理结果',
                '',
                'PNG文件 (*.png);;JPEG文件 (*.jpg *.jpeg);;BMP文件 (*.bmp)'
            )

            if file_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # 保存图像
                success = cv2.imwrite(file_path, self.result_img)

                if success:
                    QMessageBox.information(self, '成功', f'图像已保存到: {file_path}')
                    print(f'图像已保存: {file_path}')
                else:
                    QMessageBox.warning(self, '警告', '保存图像失败')

        except Exception as e:
            QMessageBox.critical(self, '错误', f'保存图像失败: {str(e)}')


def get_app_style():
    """
    获取应用程序样式表

    返回:
        样式表字符串
    """
    return '''
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #007acc;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #005a9e;
        }
        QPushButton:pressed {
            background-color: #004080;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QSlider::groove:horizontal {
            height: 8px;
            background: #ddd;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #007acc;
            width: 18px;
            height: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QComboBox {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 3px;
            min-height: 30px;
        }
    '''
