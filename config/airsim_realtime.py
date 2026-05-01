"""
AirSim 实时传感器融合定位
连接 AirSim 无人机，实时读取 IMU，运行 Mahony + EKF，输出轨迹与 RMSE
"""

import airsim
import numpy as np
import pandas as pd
import time
import os
import sys

# 添加 src 路径，以便调用你的算法模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mahony_ahrs import mahony_ahrs
from ekf_fusion import state_predict, state_jacobian, observe_state, observe_jacobian, quat_norm

# ====================== 配置 ======================
DT = 1.0 / 200.0          # 200Hz
GRAVITY = 9.81
KP = 1.0
KI = 0.05

# EKF 噪声参数（先用默认值，后续可从 Allan 分析导入）
Q = np.eye(16) * 0.001
R_obs = np.eye(6) * 0.01

# 输出路径
OUTPUT_DIR = r"D:\同济\airsim_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 连接 AirSim ======================
print("正在连接 AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("已连接 ✓")

# 开启 API 控制并起飞
client.enableApiControl(True)
client.takeoffAsync().join()
print("起飞完成")

# 可选：设置风场测试鲁棒性
# client.simSetWind(airsim.Vector3r(3, 0, 0))  # 3m/s 侧风

# ====================== 初始化状态 ======================
# Mahony 初始状态
q_mahony = np.array([1.0, 0.0, 0.0, 0.0])
bias_mahony = np.zeros(3)

# EKF 初始状态
x_ekf = np.zeros(16)
x_ekf[6] = 1.0
P_ekf = np.eye(16) * 0.1
P_ekf[6:10, 6:10] = np.eye(4) * 1e-6

# 数据存储
records = []

# ====================== 主循环 ======================
print("开始实时定位...")
duration = 30  # 运行30秒
start_time = time.time()

while time.time() - start_time < duration:
    t_now = time.time() - start_time
    
    # 1. 读取 AirSim IMU
    imu = client.getImuData()
    gyro = np.array([imu.angular_velocity.x_val,
                     imu.angular_velocity.y_val,
                     imu.angular_velocity.z_val])
    accel = np.array([imu.linear_acceleration.x_val,
                      imu.linear_acceleration.y_val,
                      imu.linear_acceleration.z_val])
    
    # 2. 读取 AirSim Ground Truth
    pose = client.simGetVehiclePose()
    gt_pos = np.array([pose.position.x_val,
                       pose.position.y_val,
                       pose.position.z_val])
    gt_q = np.array([pose.orientation.w_val,
                     pose.orientation.x_val,
                     pose.orientation.y_val,
                     pose.orientation.z_val])
    
    # 3. Mahony 姿态解算（调用你已有的函数）
    q_mahony, bias_mahony = mahony_ahrs(gyro, accel, q_mahony, bias_mahony, DT, KP, KI)
    
    # 4. EKF 预测步骤（调用你已有的函数）
    x_pred = state_predict(x_ekf, gyro, accel, DT)
    F = state_jacobian(x_ekf, gyro, accel, DT)
    P_pred = F @ P_ekf @ F.T + Q
    
    # 5. EKF 更新步骤（用 AirSim GT 做观测，模拟视觉定位）
    z = np.concatenate([gt_pos, np.zeros(3)])  # 简化：直接观测位置
    H = observe_jacobian(x_pred)
    z_pred = observe_state(x_pred)
    y = z - z_pred
    S = H @ P_pred @ H.T + R_obs
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_ekf = x_pred + K @ y
    P_ekf = (np.eye(16) - K @ H) @ P_pred
    x_ekf[6:10] = quat_norm(x_ekf[6:10])
    
    # 6. 记录
    records.append({
        "time": t_now,
        "ekf_px": x_ekf[0], "ekf_py": x_ekf[1], "ekf_pz": x_ekf[2],
        "gt_px": gt_pos[0], "gt_py": gt_pos[1], "gt_pz": gt_pos[2]
    })
    
    time.sleep(DT)

# ====================== 降落并保存 ======================
client.landAsync().join()
client.enableApiControl(False)
print("飞行结束")

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, "airsim_trajectory.csv"), index=False)

# 计算 RMSE
errors = df[["ekf_px","ekf_py","ekf_pz"]].values - df[["gt_px","gt_py","gt_pz"]].values
rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
print(f"AirSim 实时定位 RMSE: {rmse:.4f} m")
