# -*- coding: utf-8 -*-
"""
Retinex图像增强算法核心实现
========================================
该模块实现了多种Retinex算法，用于图像增强和颜色恢复。
Retinex理论模拟人类视觉系统对光照变化的感知能力，
能够有效改善低对比度、光照不均的图像质量。

核心算法:
1. SSR (Single Scale Retinex) - 单尺度Retinex算法
2. MSR (Multi Scale Retinex) - 多尺度Retinex算法
3. MSRCR - 带颜色恢复的多尺度Retinex算法
4. Automated MSRCR - 自动化的多尺度Retinex算法
5. MSRCP - 带颜色保持的多尺度Retinex算法

依赖库:
    - NumPy: 数值计算
    - OpenCV (cv2): 图像处理和卷积操作
"""

import numpy as np
import cv2


def singleScaleRetinex(img, sigma):
    """
    单尺度Retinex算法 (SSR)

    该算法通过单一尺度的高斯模糊估计光照分量，然后从原图中减去光照估计，
    得到反射分量，从而增强图像细节和对比度。

    算法原理:
        R(x,y) = log(S(x,y)) - log(F(x,y) * S(x,y))
        其中:
        - S(x,y): 原始图像
        - F(x,y): 高斯模糊核（光照估计）
        - R(x,y): Retinex增强后的图像（反射分量）

    参数:
        img: 输入图像（numpy数组，BGR格式）
        sigma: 高斯核的标准差，控制模糊程度
              - 较小的sigma（如15-30）：保留更多细节，增强局部对比度
              - 较大的sigma（如80-300）：增强全局对比度，但细节较少

    返回:
        retinex: 单尺度Retinex增强后的图像
    """
    # 计算Retinex: log(原图像) - log(高斯模糊后的图像)
    # 高斯模糊用于估计图像的光照分量
    retinex = np.log10(img) - np.log10(
        cv2.GaussianBlur(img, (0, 0), sigma))  # r(x,y)= logS(x,y)-log[F(x,y)*s(x,y)] SSR模拟近似L(x,y)

    return retinex


def multiScaleRetinex(img, sigma_list):
    """
    多尺度Retinex算法 (MSR)

    该算法通过融合多个不同尺度的SSR结果，兼顾局部和全局的增强效果。
    多尺度融合能够避免单一尺度的局限性，获得更好的视觉效果。

    算法原理:
        MSR = (1/n) * Σ[SSR_i]
        其中:
        - n: 尺度数量
        - SSR_i: 第i个尺度的单尺度Retinex结果

    参数:
        img: 输入图像（numpy数组，BGR格式）
        sigma_list: 高斯核标准差列表，例如 [15, 80, 250]
                   - 小尺度: 增强局部细节
                   - 中尺度: 平衡局部和全局
                   - 大尺度: 增强全局对比度

    返回:
        retinex: 多尺度加权平均后的Retinex图像
    """
    # 初始化结果矩阵
    retinex = np.zeros_like(img)

    # 对每个尺度执行SSR并累加
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    # 计算多尺度的平均值
    retinex = retinex / len(sigma_list)

    return retinex


def colorRestoration(img, alpha, beta):
    """
    颜色恢复函数 (Color Restoration, CR)

    在多尺度Retinex基础上进行颜色恢复，补偿颜色信息的损失，
    使增强后的图像保持自然的色彩。

    算法原理:
        CR_i(x,y) = β * [log(α * C_i(x,y)) - log(ΣC_j(x,y))]
        其中:
        - C_i(x,y): 第i个颜色通道的值
        - α: 颜色强度调整系数，控制颜色恢复强度
        - β: 颜色平衡系数，影响颜色平衡程度

    参数:
        img: 输入图像（numpy数组，BGR格式）
        alpha: 颜色强度调整系数（通常取值125左右）
               - 增大：增强颜色强度
               - 减小：减弱颜色强度
        beta: 颜色平衡系数（通常取值46左右）
              - 调整RGB通道之间的平衡关系

    返回:
        color_restoration: 颜色恢复系数矩阵，用于与MSR结果相乘

    注意:
        keepdims=True 参数确保维度保持一致，避免广播错误
    """
    # 计算RGB三个通道的和，keepdims保持维度不变
    img_sum = np.sum(img, axis=2, keepdims=True)  # keepdims这个参数为True，被删去的维度在结果矩阵中就被设置为一。

    # 计算颜色恢复系数
    # beta控制整体强度，alpha控制单通道强度
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):
    """
    简单颜色平衡算法

    通过直方图裁剪来限制像素值范围，消除极端的亮暗区域，
    防止图像过曝或欠曝，增强整体对比度。

    算法原理:
        1. 统计每个通道的像素值分布
        2. 根据low_clip和high_clip确定裁剪阈值
        3. 将低于下限的值提升到下限，高于上限的值降低到上限

    参数:
        img: 输入图像（numpy数组，BGR格式）
        low_clip: 低裁剪比例（0-1之间），例如0.01表示裁剪最低1%的像素
                 - 较大的值：更多的暗部被提升，增强暗部细节
                 - 较小的值：保留更多暗部信息
        high_clip: 高裁剪比例（0-1之间），例如0.99表示裁剪最高1%的像素
                  - 较小的值：更多的亮部被降低，避免过曝
                  - 较大的值：保留更多亮部信息

    返回:
        img: 颜色平衡后的图像
    """
    # 计算总像素数
    total = img.shape[0] * img.shape[1]

    # 对每个颜色通道（BGR）分别处理
    for i in range(img.shape[2]):
        # 获取该通道中所有唯一的像素值及其出现次数
        unique, counts = np.unique(img[:, :, i], return_counts=True)

        # 累计像素计数，确定裁剪阈值
        current = 0
        for u, c in zip(unique, counts):
            # 如果累计比例低于low_clip，更新低阈值
            if float(current) / total < low_clip:
                low_val = u
            # 如果累计比例低于high_clip，更新高阈值
            if float(current) / total < high_clip:
                high_val = u
            current += c

        # 应用裁剪：限制像素值在[low_val, high_val]范围内
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    """
    带颜色恢复的多尺度Retinex算法 (MSRCR)

    结合MSR和颜色恢复，能够同时增强图像对比度和恢复自然色彩，
    是最常用的Retinex算法之一。

    算法流程:
        1. 执行多尺度Retinex (MSR)
        2. 计算颜色恢复系数 (CR)
        3. 将MSR与CR结合，应用增益和偏差
        4. 归一化到[0, 255]范围
        5. 应用颜色平衡

    参数:
        img: 输入图像（numpy数组，BGR格式）
        sigma_list: 多尺度高斯标准差列表
        G: 增益系数（Gain），控制整体亮度（通常取值192左右）
           - 增大：图像变亮
           - 减小：图像变暗
        b: 偏差系数（Bias），控制整体对比度（通常取值-30左右）
          - 负值：提高对比度
          - 正值：降低对比度
        alpha: 颜色强度调整系数（见colorRestoration函数）
        beta: 颜色平衡系数（见colorRestoration函数）
        low_clip: 低裁剪比例（见simplestColorBalance函数）
        high_clip: 高裁剪比例（见simplestColorBalance函数）

    返回:
        img_msrcr: MSRCR增强后的图像（uint8类型，范围0-255）
    """
    # 转换为float64类型并加1，避免log(0)错误
    img = np.float64(img) + 1.0  # 保证精度

    # 执行多尺度Retinex，得到增强后的反射分量
    img_retinex = multiScaleRetinex(img, sigma_list)

    # 计算颜色恢复系数
    img_color = colorRestoration(img, alpha, beta)

    # 结合Retinex结果和颜色恢复，应用增益G和偏差b
    # 公式: result = G * (MSR * CR + b)
    img_msrcr = G * (img_retinex * img_color + b)  # G 增益 b偏差

    # 对每个通道进行归一化处理到[0, 255]范围
    for i in range(img_msrcr.shape[2]):  # 归一化处理
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / (
                np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * 255

    # 裁剪到[0, 255]范围并转换为uint8类型
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))

    # 应用颜色平衡，进一步优化图像质量
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

    return img_msrcr


def automatedMSRCR(img, sigma_list):
    """
    自动化多尺度Retinex算法 (Automated MSRCR)

    自动调整参数的MSRCR算法，无需手动配置颜色恢复和增益系数。
    通过统计像素分布自适应地确定裁剪范围。

    算法特点:
        - 自动化参数选择，简化使用流程
        - 适用于不同类型的图像
        - 保持较好的颜色自然度

    参数:
        img: 输入图像（numpy数组，BGR格式）
        sigma_list: 多尺度高斯标准差列表

    返回:
        img_retinex: 自动化MSRCR增强后的图像（uint8类型，范围0-255）

    算法流程:
        1. 执行多尺度Retinex
        2. 对每个通道统计零值像素数量
        3. 根据零值像素比例确定裁剪范围
        4. 归一化到[0, 255]范围
    """
    # 转换为float64类型并加1
    img = np.float64(img) + 1.0

    # 执行多尺度Retinex
    img_retinex = multiScaleRetinex(img, sigma_list)

    # 对每个通道分别处理
    for i in range(img_retinex.shape[2]):
        # 统计像素值分布（乘以100转为整数便于计数）
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)

        # 查找零值像素数量
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        # 初始裁剪范围设为最小和最大值
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0

        # 根据零值像素比例自适应调整裁剪范围
        # 寻找两侧分布稀疏的区域作为裁剪边界
        for u, c in zip(unique, count):
            # 低端：负值且出现次数少于零值10%
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            # 高端：正值且出现次数少于零值10%
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        # 应用自适应裁剪范围
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        # 归一化到[0, 255]范围
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    # 转换为uint8类型
    img_retinex = np.uint8(img_retinex)

    return img_retinex


def MSRCP(img, sigma_list, low_clip, high_clip):
    """
    带颜色保持的多尺度Retinex算法 (MSRCP)

    在增强图像对比度的同时保持原始RGB通道的颜色比例关系，
    适用于需要保持原始颜色特征的场景。

    算法原理:
        1. 计算图像的亮度通道（RGB平均值）
        2. 对亮度通道执行多尺度Retinex
        3. 根据增强前后的亮度比例调整RGB各通道
        4. 保持原始RGB通道的比例关系不变

    参数:
        img: 输入图像（numpy数组，BGR格式）
        sigma_list: 多尺度高斯标准差列表
        low_clip: 低裁剪比例（见simplestColorBalance函数）
        high_clip: 高裁剪比例（见simplestColorBalance函数）

    返回:
        img_msrcp: MSRCP增强后的图像（uint8类型，范围0-255）

    算法特点:
        - 保持原始颜色比例，颜色偏移较小
        - 增强对比度同时避免颜色失真
        - 适用于颜色信息重要的场景（如医学影像、艺术作品等）
    """
    # 转换为float64类型并加1
    img = np.float64(img) + 1.0

    # 计算亮度通道（RGB三个通道的平均值）
    intensity = np.sum(img, axis=2) / img.shape[2]

    # 对亮度通道执行多尺度Retinex增强
    retinex = multiScaleRetinex(intensity, sigma_list)

    # 扩展维度以匹配原始图像的通道结构
    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    # 对增强后的亮度应用颜色平衡
    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    # 直接线性量化到[1, 256]范围（避免0值）
    # 直接线性量化
    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    # 初始化结果图像
    img_msrcp = np.zeros_like(img)

    # 根据原始的RGB的比例映射到每个通道
    # 对每个像素，保持RGB通道的比例关系，同时应用亮度增强
    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            # 获取当前像素RGB通道的最大值
            B = np.max(img[y, x])#在这一行，计算当前像素位置(y, x)处的图像img的所有通道的最大值，并将这个最大值赋给变量B

            # 计算映射比例A
            # 确保增强后的RGB值不超过256，同时应用亮度增强比例
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])#映射比例A

            # 应用映射比例到每个RGB通道，保持原始比例关系
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    # 转换到[0, 255]范围的uint8类型
    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp
