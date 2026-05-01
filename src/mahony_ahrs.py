import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import os

# ----- 输入文件 -----
IMU_CSV = os.path.join(r"D:\同济\euroc_data\数据集\预处理.csv")
NOISE_JSON = os.path.join(r"D:\同济\euroc_data\数据集\allan_results", "imu_noise_params.json")
GT_CSV = os.path.join(r"D:\同济\euroc_data\数据集\MH_01_easy\mav0\state_groundtruth_estimate0\data.csv")  # ← 已修正为你刚提取的GT
OUTPUT_DIR = os.path.join(r"D:\同济\euroc_data\数据集\mahony_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 2. Mahony 滤波器参数 ======================
FREQ = 200.0                  # IMU频率 (Hz)
DT = 1.0 / FREQ
KP = 1.0                      # 比例增益（加速度计修正强度）
KI = 0.05                     # 积分增益（陀螺仪偏置修正）
GRAVITY = 9.81

# ====================== 3. Mahony 核心算法 ======================
def mahony_ahrs(gyro, accel, q, bias, dt, kp, ki):
    """
    Mahony 互补滤波姿态解算
    参数:
        gyro:  角速度 [gx, gy, gz] (rad/s)
        accel: 加速度 [ax, ay, az] (m/s²) —— 已重力补偿
        q:     当前四元数 [qw, qx, qy, qz]
        bias:  陀螺仪偏置估计 [bx, by, bz]
        dt:    采样间隔 (s)
        kp, ki: PI 增益
    返回:
        q_new:   更新后的四元数
        bias_new: 更新后的偏置估计
    """
    qw, qx, qy, qz = q

    # 1) 加速度归一化
    acc_norm = np.linalg.norm(accel)
    if acc_norm < 1e-10:
        # 加速度太小，不做修正
        q_dot = 0.5 * np.array([
            -qx*gyro[0] - qy*gyro[1] - qz*gyro[2],
             qw*gyro[0] + qy*gyro[2] - qz*gyro[1],
             qw*gyro[1] - qx*gyro[2] + qz*gyro[0],
             qw*gyro[2] + qx*gyro[1] - qy*gyro[0]
        ])
        q_new = q + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new)
        return q_new, bias

    acc = accel / acc_norm

    # 2) 重力方向在机体坐标系的理论值
    # 从四元数推导：机体坐标系下重力方向 = R^T * [0,0,1]
    vx = 2 * (qx*qz - qw*qy)
    vy = 2 * (qy*qz + qw*qx)
    vz = 1 - 2 * (qx*qx + qy*qy)

    # 3) 误差 = 加速度测量值 × 重力理论值（叉积）
    ex = acc[1]*vz - acc[2]*vy
    ey = acc[2]*vx - acc[0]*vz
    ez = acc[0]*vy - acc[1]*vx

    # 4) PI 控制器 → 修正角速度
    bias = bias + np.array([ex, ey, ez]) * ki * dt
    gyro_corrected = gyro + np.array([ex, ey, ez]) * kp + bias

    # 5) 四元数更新
    gx, gy, gz = gyro_corrected
    q_dot = 0.5 * np.array([
        -qx*gx - qy*gy - qz*gz,
         qw*gx + qy*gz - qz*gy,
         qw*gy - qx*gz + qz*gx,
         qw*gz + qx*gy - qy*gx
    ])

    q_new = q + q_dot * dt
    q_new = q_new / np.linalg.norm(q_new)

    return q_new, bias

# ====================== 4. 加载数据 ======================
print("=" * 60)
print("Mahony 姿态解算")
print("=" * 60)

df_imu = pd.read_csv(IMU_CSV)
print(f"IMU数据: {len(df_imu)} 行")

# 加载 GT 姿态
df_gt = None
GT_CSV_PATH = os.path.join(r"D:\同济\euroc_data\数据集\MH_01_easy\mav0\state_groundtruth_estimate0\data.csv")
if os.path.exists(GT_CSV_PATH):
    # 先读第一行看列名
    df_gt = pd.read_csv(GT_CSV_PATH)
    print(f"GT 实际列名: {df_gt.columns.tolist()[:5]}...")
    
    # 如果列名是数字（说明原始列名被当成数据了），手动指定
    if 'qw' not in df_gt.columns and 'q_RS_w []' not in df_gt.columns:
        # 关闭文件重新读，跳过第一行
        GT_COLS = ["ts_ns", "px","py","pz","qw","qx","qy","qz",
                   "vx","vy","vz","bwx","bwy","bwz","bax","bay","baz"]
        df_gt = pd.read_csv(GT_CSV_PATH, skiprows=1, names=GT_COLS)
        # 转换时间戳
        if df_gt["ts_ns"].iloc[0] > 1e10:
            df_gt["ts"] = df_gt["ts_ns"] * 1e-9
        else:
            df_gt["ts"] = df_gt["ts_ns"]
    elif 'q_RS_w []' in df_gt.columns:
        # EuRoC 原始列名
        df_gt = df_gt.rename(columns={
            'q_RS_w []': 'qw', 'q_RS_x []': 'qx',
            'q_RS_y []': 'qy', 'q_RS_z []': 'qz',
            'p_RS_R_x [m]': 'px', 'p_RS_R_y [m]': 'py', 'p_RS_R_z [m]': 'pz'
        })
    
    print(f"GT数据: {len(df_gt)} 行")
    print(f"GT列名(修正后): {df_gt.columns.tolist()[:5]}...")
else:
    print("⚠ 无Ground Truth，仅输出姿态")

# ====================== 5. 运行 Mahony ======================
n = len(df_imu)
q = np.array([1.0, 0.0, 0.0, 0.0])  # 初始四元数
bias = np.zeros(3)

quaternions = np.zeros((n, 4))
euler_angles = np.zeros((n, 3))
bias_history = np.zeros((n, 3))

for i in range(n):
    gyro = df_imu[["wx_raw", "wy_raw", "wz_raw"]].iloc[i].values
    accel = df_imu[["ax_compensated", "ay_compensated", "az_compensated"]].iloc[i].values

    q, bias = mahony_ahrs(gyro, accel, q, bias, DT, KP, KI)

    quaternions[i] = q
    euler_angles[i] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz', degrees=True)
    bias_history[i] = bias

# ====================== 6. 与 GT 对比 ======================
if df_gt is not None:
    # 时间对齐（简化：按行数比例对齐）
    gt_step = len(df_gt) / len(df_imu)
    gt_indices = (np.arange(n) * gt_step).astype(int)
    gt_indices = np.clip(gt_indices, 0, len(df_gt)-1)

    gt_quat = df_gt[["qw","qx","qy","qz"]].values[gt_indices]
    gt_euler = np.array([Rot.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz', degrees=True)
                         for q in gt_quat])

    # RMSE 姿态误差
    error_euler = euler_angles - gt_euler
    # 处理角度环绕
    error_euler = np.arctan2(np.sin(np.deg2rad(error_euler)), np.cos(np.deg2rad(error_euler)))
    error_euler = np.rad2deg(error_euler)

    rmse_roll = np.sqrt(np.mean(error_euler[:, 0]**2))
    rmse_pitch = np.sqrt(np.mean(error_euler[:, 1]**2))
    rmse_yaw = np.sqrt(np.mean(error_euler[:, 2]**2))

    print(f"\n========== 姿态精度 (Mahony vs GT) ==========")
    print(f"RMSE Roll:  {rmse_roll:.4f}°")
    print(f"RMSE Pitch: {rmse_pitch:.4f}°")
    print(f"RMSE Yaw:   {rmse_yaw:.4f}°")
    print(f"RMSE 平均:  {np.mean([rmse_roll, rmse_pitch, rmse_yaw]):.4f}°")

# ====================== 7. 可视化 ======================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

time = np.arange(n) * DT

# 子图1：欧拉角
ax1 = axes[0, 0]
ax1.plot(time, euler_angles[:, 0], 'r-', lw=0.8, label='Mahony Roll')
ax1.plot(time, euler_angles[:, 1], 'g-', lw=0.8, label='Mahony Pitch')
ax1.plot(time, euler_angles[:, 2], 'b-', lw=0.8, label='Mahony Yaw')
if df_gt is not None:
    ax1.plot(time, gt_euler[:, 0], 'r--', lw=0.5, alpha=0.5, label='GT Roll')
    ax1.plot(time, gt_euler[:, 1], 'g--', lw=0.5, alpha=0.5, label='GT Pitch')
    ax1.plot(time, gt_euler[:, 2], 'b--', lw=0.5, alpha=0.5, label='GT Yaw')
ax1.set_xlabel("时间 (s)")
ax1.set_ylabel("角度 (°)")
ax1.set_title("Mahony 姿态估计")
ax1.legend(fontsize=7)
ax1.grid(alpha=0.3)

# 子图2：姿态误差
if df_gt is not None:
    ax2 = axes[0, 1]
    ax2.plot(time, error_euler[:, 0], 'r-', lw=0.8, label='Roll 误差')
    ax2.plot(time, error_euler[:, 1], 'g-', lw=0.8, label='Pitch 误差')
    ax2.plot(time, error_euler[:, 2], 'b-', lw=0.8, label='Yaw 误差')
    ax2.axhline(y=0, color='k', ls=':', alpha=0.3)
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("误差 (°)")
    ax2.set_title(f"姿态误差 (RMSE: {rmse_roll:.2f}°/{rmse_pitch:.2f}°/{rmse_yaw:.2f}°)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, '无GT数据', transform=axes[0,1].transAxes, ha='center', fontsize=14)

# 子图3：陀螺仪偏置估计
ax3 = axes[1, 0]
ax3.plot(time, bias_history[:, 0], 'r-', lw=0.8, label='Bias X')
ax3.plot(time, bias_history[:, 1], 'g-', lw=0.8, label='Bias Y')
ax3.plot(time, bias_history[:, 2], 'b-', lw=0.8, label='Bias Z')
ax3.set_xlabel("时间 (s)")
ax3.set_ylabel("偏置 (rad/s)")
ax3.set_title("Mahony 陀螺仪偏置估计")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# 子图4：四元数
ax4 = axes[1, 1]
ax4.plot(time, quaternions[:, 0], 'k-', lw=0.8, label='qw')
ax4.plot(time, quaternions[:, 1], 'r-', lw=0.8, label='qx')
ax4.plot(time, quaternions[:, 2], 'g-', lw=0.8, label='qy')
ax4.plot(time, quaternions[:, 3], 'b-', lw=0.8, label='qz')
ax4.set_xlabel("时间 (s)")
ax4.set_ylabel("四元数值")
ax4.set_title("Mahony 四元数")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.suptitle("Mahony 互补滤波姿态解算 (PI: Kp=%.1f, Ki=%.2f)" % (KP, KI), fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mahony_attitude.png"), dpi=150)
plt.show()

# ====================== 8. 保存结果 ======================
df_out = pd.DataFrame({
    "time": time,
    "qw": quaternions[:, 0], "qx": quaternions[:, 1],
    "qy": quaternions[:, 2], "qz": quaternions[:, 3],
    "roll": euler_angles[:, 0], "pitch": euler_angles[:, 1], "yaw": euler_angles[:, 2],
    "bias_x": bias_history[:, 0], "bias_y": bias_history[:, 1], "bias_z": bias_history[:, 2]
})

out_csv = os.path.join(OUTPUT_DIR, "mahony_attitude.csv")
df_out.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\n结果已保存至: {out_csv}")
print("Mahony 姿态解算完成！")
