# -*- coding: utf-8 -*-
"""
数字图像处理大作业 —— 改进版（Improved）
==========================================
在基础版 enhance_baseline.py 的基础上做两处关键修改：

【修改1】颜色空间：不再直接对 BGR 三通道分别均衡化，
         而是在 HSV 空间只对亮度通道 V 做增强，
         色调 H、饱和度 S 保持不变 —— 解决颜色失真问题。

【修改2】增强算法：用 “CLAHE（限制对比度自适应直方图均衡化）
         + 伽马提亮” 替代全局直方图均衡化。
         - CLAHE：分块(tile)自适应均衡化，照顾局部对比度，
           并通过 clipLimit 限制对比度，抑制噪声放大和局部过增强；
         - 伽马提亮(γ=0.45)：在 CLAHE 之后整体提亮，
           把亮度恢复到接近正常光照水平。

运行：python enhance_improved.py
依赖：opencv-python、numpy
"""
import os
import cv2
import numpy as np

# 复用基础版的数据准备（低光照模拟），保证两版输入完全一致、可公平对比
from enhance_baseline import make_low_light, _gamma_table


def enhance_improved(img_bgr, clip_limit=2.0, tile_grid=(8, 8), bright_gamma=0.45):
    """
    改进版增强：
    1) BGR -> HSV；
    2) 对 V 通道先做 CLAHE（局部对比度、限制噪声）；
    3) 再做伽马提亮（γ<1，整体恢复亮度）；
    4) H、S 不变，合并后 HSV -> BGR。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 第一步：CLAHE 局部对比度增强（clipLimit 限制对比度，tileGridSize 分块）
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    v = clahe.apply(v)

    # 第二步：伽马提亮，恢复整体亮度
    v = cv2.LUT(v, _gamma_table(bright_gamma))

    hsv_eq = cv2.merge((h, s, v))                 # H、S 原样保留
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    img = cv2.imread(os.path.join(here, "data", "lena.png"))
    low = make_low_light(img)
    enhanced = enhance_improved(low)
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "improved_lena_lowlight.png"), low)
    cv2.imwrite(os.path.join(out_dir, "improved_lena_enhanced.png"), enhanced)
    print("[Improved] 已保存改进版结果到 results/ 目录")


if __name__ == "__main__":
    main()
