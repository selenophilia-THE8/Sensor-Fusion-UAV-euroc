# Sensor-Fusion-UAV

**基于多传感器数据融合的无人机基础定位精度提升研究**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

面向低空悬停、近距离飞行场景，构建**“时序对齐 → Allan噪声建模 → Mahony姿态解算 → EKF融合估计”**四层融合架构，实现IMU与视觉的高低频互补定位。

**核心指标**：EuRoC MH_01 序列定位 RMSE **0.12米**。

## 架构图
EuRoC原始数据
↓ 预处理
预处理CSV
↓ Allan方差分析
IMU噪声参数 (ARW, VRW, BI)
↓ EKF融合
定位轨迹 (RMSE 0.12m)
↓ Mahony解算
姿态估计 (对照实验)

## 项目结构
Sensor-Fusion-UAV/
├── config/
│ ├── euroc_config.yaml ← EuRoC 数据集配置
│ └── noise_params.json ← Allan 噪声参数
├── src/
│ ├── preprocessing.py ← 数据预处理
│ ├── allan_noise.py ← Allan 方差噪声分析
│ ├── ekf_fusion.py ← EKF 融合定位
│ └── mahony_ahrs.py ← Mahony 姿态解算
├── scripts/
│ └── run_full_pipeline.py ← 一键运行流水线
├── data/
├── output/
└── README.md

## 快速开始

```bash
pip install -r requirements.txt
python scripts/run_full_pipeline.py --dataset MH_01_easy
