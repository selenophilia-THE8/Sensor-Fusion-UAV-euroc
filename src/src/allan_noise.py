# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:35:55 2026

@author: dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ====================== 1. 基础配置 ======================
# 输入文件（你预处理后的数据）
INPUT_CSV = r"D:\同济\euroc_data\数据集\预处理.csv"

# 输出目录（保存Allan方差图和噪声参数）
OUTPUT_DIR = r"D:\同济\euroc_data\数据集\allan_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IMU采样频率（EuRoC IMU是200Hz）
IMU_FREQ = 200.0  # Hz
DT = 1.0 / IMU_FREQ  # 采样间隔

# 待分析的传感器列
GYRO_COLS = ["wx_raw", "wy_raw", "wz_raw"]       # 角速度（rad/s）
ACCEL_COLS = ["ax_raw", "ay_raw", "az_raw"]      # 加速度（m/s²）

# ====================== 2. 加载数据 ======================
print("=" * 60)
print("正在加载预处理数据...")
df = pd.read_csv(INPUT_CSV)

# 检查必需列是否存在
required_cols = ["relative_time"] + GYRO_COLS + ACCEL_COLS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺少列: {missing}")

print(f"数据行数: {len(df)}")
print(f"时间范围: {df['relative_time'].min():.2f}s ~ {df['relative_time'].max():.2f}s")
print(f"预计IMU频率: {1.0 / (df['relative_time'].iloc[1] - df['relative_time'].iloc[0]):.1f} Hz")

# ====================== 3. Allan方差核心函数 ======================
def allan_variance(signal, dt):
    """
    计算单个传感器信号的Allan方差
    参数:
        signal: 1D numpy数组，传感器信号（角速度/加速度）
        dt: 采样间隔（秒）
    返回:
        tau: 平均时间簇（秒）
        sigma: Allan标准差（与输入信号同单位）
    """
    n = len(signal)
    
    # 最大簇数（取整数，最多到信号长度的1/9以保证统计显著性）
    max_clusters = int(np.floor(n / 9))
    
    # 生成tau序列（对数均匀分布，保证曲线光滑）
    tau_list = []
    m = 1
    while m <= max_clusters:
        tau_list.append(m)
        if m < 10:
            m += 1
        else:
            m = int(m * 1.1)  # 逐步增大步长
    tau_list = np.unique(tau_list)  # 去重
    
    sigma_list = []
    for m in tau_list:
        # 分成K簇
        K = n // m
        if K < 2:
            continue
        
        # 每簇取平均值
        clusters = signal[:K * m].reshape(K, m)
        means = np.mean(clusters, axis=1)
        
        # Allan方差 = 1/2 * <(θ_k+1 - θ_k)²>
        # 对于角速度/加速度信号，我们算的是rate Allan variance
        diff = np.diff(means)
        sigma_sq = 0.5 * np.mean(diff**2)
        sigma = np.sqrt(sigma_sq)
        
        sigma_list.append(sigma)
    
    tau = np.array([dt * t for t in tau_list[:len(sigma_list)]])
    sigma = np.array(sigma_list)
    
    return tau, sigma

def fit_allan_noise(tau, sigma, sensor_type="gyro"):
    """
    从Allan标准差曲线拟合噪声参数
    参数:
        tau: 平均时间（秒）
        sigma: Allan标准差
        sensor_type: "gyro" 或 "accel"
    返回:
        noise_params: 字典，包含ARW, BI, RRW等
    """
    noise_params = {}
    
    # ---- 角度随机游走 (ARW) / 加速度随机游走 (VRW) ----
    # 斜率 = -1/2 区域: σ(τ) = N / sqrt(τ)
    # 拟合 τ = 1s 附近的值
    target_tau = 1.0
    idx_1s = np.argmin(np.abs(tau - target_tau))
    
    # 取 τ < 10s 且 σ 单调递减的区域（避开最低点之后）
    fit_mask = (tau < 10.0) & (sigma == np.minimum.accumulate(sigma[::-1])[::-1])
    # 简单方法：取前20%的tau点
    n_fit = max(3, len(tau) // 5)
    
    if n_fit >= 3:
        log_tau_fit = np.log(tau[:n_fit])
        log_sigma_fit = np.log(sigma[:n_fit])
        
        # 线性拟合: log(σ) = log(N) - 0.5*log(τ)
        coeffs = np.polyfit(log_tau_fit, log_sigma_fit, 1)
        N = np.exp(coeffs[1])  # 截距
        
        if sensor_type == "gyro":
            noise_params["ARW"] = N  # rad/s/√Hz → deg/√hr 可以额外转换
            noise_params["ARW_deg_per_sqrt_hr"] = N * 180 / np.pi * 60  # 转换为 deg/√hr
        else:
            noise_params["VRW"] = N  # m/s²/√Hz
            noise_params["VRW_ug_per_sqrt_Hz"] = N * 1e6 / 9.81  # 转换为 μg/√Hz
    
    # ---- 偏置不稳定性 (BI) ----
    # 寻找Allan曲线的最低点（斜率=0区域）
    min_idx = np.argmin(sigma[:len(sigma)//2])  # 只在前半段找（避免速率斜坡干扰）
    
    if min_idx > 0:
        sigma_min = sigma[min_idx]
        tau_min = tau[min_idx]
        
        # BI = σ_min / 0.664（标准转换因子）
        BI_value = sigma_min / 0.664
        
        if sensor_type == "gyro":
            noise_params["BI"] = BI_value  # rad/s
            noise_params["BI_deg_per_hr"] = BI_value * 180 / np.pi * 3600  # °/hr
        else:
            noise_params["BI"] = BI_value  # m/s²
            noise_params["BI_ug"] = BI_value * 1e6 / 9.81  # μg
        
        noise_params["BI_tau"] = tau_min  # BI对应的时间常数
    
    # ---- 速率随机游走 (RRW) ----
    # 斜率 = +1/2 区域: σ(τ) = K * sqrt(τ/3)
    if min_idx < len(tau) - 3:
        rr_region = tau > tau[min_idx] * 3  # BI之后3倍区域
        if np.sum(rr_region) >= 3:
            log_tau_rr = np.log(tau[rr_region])
            log_sigma_rr = np.log(sigma[rr_region])
            coeffs_rr = np.polyfit(log_tau_rr, log_sigma_rr, 1)
            K = np.exp(coeffs_rr[1] - 0.5 * np.log(3))
            
            if sensor_type == "gyro":
                noise_params["RRW"] = K  # rad/s/√s
            else:
                noise_params["RRW"] = K  # m/s²/√s
    
    return noise_params

# ====================== 4. 逐轴计算Allan方差 ======================
print("\n" + "=" * 60)
print("正在计算Allan方差...")

all_results = {}

for sensor_type, cols in [("gyro", GYRO_COLS), ("accel", ACCEL_COLS)]:
    for col in cols:
        print(f"\n处理: {col}")
        
        # 提取信号
        signal = df[col].values
        
        # 去除均值（零均值化）
        signal = signal - np.mean(signal)
        
        # 计算Allan方差
        tau, sigma = allan_variance(signal, DT)
        
        # 拟合噪声参数
        params = fit_allan_noise(tau, sigma, sensor_type)
        
        all_results[col] = {
            "tau": tau,
            "sigma": sigma,
            "params": params
        }
        
        # 打印结果
        print(f"  --- {col} 噪声参数 ---")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")

# ====================== 5. 可视化 ======================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 陀螺仪三轴子图
gyro_axes = axes[0]
for i, (col, ax) in enumerate(zip(GYRO_COLS, gyro_axes)):
    tau = all_results[col]["tau"]
    sigma = all_results[col]["sigma"]
    params = all_results[col]["params"]
    
    ax.loglog(tau, sigma, 'b-', linewidth=1.2, label='Allan标准差')
    
    # 标注ARW (斜率=-1/2参考线)
    if "ARW" in params:
        arw_line = params["ARW"] / np.sqrt(tau)
        ax.loglog(tau, arw_line, 'r--', linewidth=1.0, alpha=0.6, label=f'ARW={params["ARW"]:.2e}')
    
    # 标注BI（最低点）
    if "BI" in params and "BI_tau" in params:
        ax.axhline(y=params["BI"] * 0.664, color='orange', linestyle='--', alpha=0.6)
        ax.annotate(f'BI={params.get("BI_deg_per_hr", params["BI"]):.2e}', 
                     xy=(params["BI_tau"], params["BI"] * 0.664),
                     xytext=(params["BI_tau"]*5, params["BI"] * 3),
                     arrowprops=dict(arrowstyle='->', alpha=0.6),
                     fontsize=8)
    
    axis_name = col.replace("_raw", "")
    ax.set_xlabel("τ（秒）")
    ax.set_ylabel("σ（rad/s）" if "w" in col else "σ（m/s²）")
    ax.set_title(f"Allan方差 - {axis_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

# 加速度计三轴子图
accel_axes = axes[1]
for i, (col, ax) in enumerate(zip(ACCEL_COLS, accel_axes)):
    tau = all_results[col]["tau"]
    sigma = all_results[col]["sigma"]
    params = all_results[col]["params"]
    
    ax.loglog(tau, sigma, 'g-', linewidth=1.2, label='Allan标准差')
    
    # 标注VRW
    if "VRW" in params:
        vrw_line = params["VRW"] / np.sqrt(tau)
        ax.loglog(tau, vrw_line, 'r--', linewidth=1.0, alpha=0.6, label=f'VRW={params["VRW"]:.2e}')
    
    # 标注BI
    if "BI" in params and "BI_tau" in params:
        ax.axhline(y=params["BI"] * 0.664, color='orange', linestyle='--', alpha=0.6)
        ax.annotate(f'BI={params.get("BI_ug", params["BI"]):.2e}', 
                     xy=(params["BI_tau"], params["BI"] * 0.664),
                     xytext=(params["BI_tau"]*5, params["BI"] * 3),
                     arrowprops=dict(arrowstyle='->', alpha=0.6),
                     fontsize=8)
    
    axis_name = col.replace("_raw", "")
    ax.set_xlabel("τ（秒）")
    ax.set_ylabel("σ（rad/s）" if "w" in col else "σ（m/s²）")
    ax.set_title(f"Allan方差 - {axis_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

plt.suptitle("EuRoC IMU Allan方差分析\n（六轴传感器噪声建模）", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "allan_variance_all_axes.png"), dpi=150, bbox_inches='tight')
plt.show()

# ====================== 6. 汇总所有噪声参数 ======================
print("\n" + "=" * 60)
print("========== 噪声参数汇总表 ==========")
print("=" * 60)

summary = []
for col in GYRO_COLS + ACCEL_COLS:
    params = all_results[col]["params"]
    row = {"轴": col.replace("_raw", "")}
    row.update({k: f"{v:.4e}" if isinstance(v, float) else v for k, v in params.items()})
    summary.append(row)

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))

# 保存汇总表
summary_csv = os.path.join(OUTPUT_DIR, "noise_parameters_summary.csv")
df_summary.to_csv(summary_csv, index=False, encoding="utf-8")
print(f"\n噪声参数汇总表已保存至: {summary_csv}")

# ====================== 7. 保存为JSON（供后续EKF直接调用） ======================
import json

noise_params_for_ekf = {}
for col in GYRO_COLS + ACCEL_COLS:
    params = all_results[col]["params"]
    clean_params = {k: v for k, v in params.items() if isinstance(v, (int, float))}
    noise_params_for_ekf[col] = clean_params

json_path = os.path.join(OUTPUT_DIR, "imu_noise_params.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(noise_params_for_ekf, f, indent=2, ensure_ascii=False)
print(f"噪声参数JSON（供EKF调用）已保存至: {json_path}")

print("\n" + "=" * 60)
print("Allan方差分析全部完成！")
print("=" * 60)
