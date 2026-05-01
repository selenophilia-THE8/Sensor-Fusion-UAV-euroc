"""
一键运行完整传感器融合定位流水线
用法: python scripts/run_full_pipeline.py --dataset MH_01_easy
"""

import os
import sys
import argparse

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    parser = argparse.ArgumentParser(description='传感器融合定位流水线')
    parser.add_argument('--dataset', type=str, default='MH_01_easy',
                        help='数据集名称')
    args = parser.parse_args()

    print(f"========== 传感器融合定位流水线 ==========")
    print(f"数据集: {args.dataset}")
    print(f"步骤: 预处理 → Allan噪声建模 → EKF融合 → Mahony姿态解算")
    print(f"=" * 60)

    # 步骤1: 预处理
    print("\n[1/4] 数据预处理...")
    os.system(f"python ../src/适宜euroc数据集的预处理.py")

    # 步骤2: Allan方差
    print("\n[2/4] Allan方差噪声分析...")
    os.system(f"python ../src/适宜euroc数据集的Allan方差噪声建模.py")

    # 步骤3: EKF融合
    print("\n[3/4] EKF融合定位...")
    os.system(f"python ../src/适宜euroc数据集的ekf融合估计.py")

    # 步骤4: Mahony（对照）
    print("\n[4/4] Mahony姿态解算（对照实验）...")
    os.system(f"python ../src/适宜euroc数据集的mahony滤波对比.py")

    print(f"\n========== 流水线完成 ==========")

if __name__ == "__main__":
    main()
