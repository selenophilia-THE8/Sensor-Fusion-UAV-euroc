import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import os
import sys

# ====================== 全局常量 ======================
FREQ = 200.0
DT = 1.0 / FREQ
KP = 1.0          # 比例增益
KI = 0.05         # 积分增益
GRAVITY = 9.81

GT_COLS_RAW = [
    "timestamp_ns", "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
    "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []",
    "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]",
    "b_w_RS_S_x [rad s^-1]", "b_w_RS_S_y [rad s^-1]", "b_w_RS_S_z [rad s^-1]",
    "b_a_RS_S_x [m s^-2]", "b_a_RS_S_y [m s^-2]", "b_a_RS_S_z [m s^-2]"
]


# ====================== Mahony 算法 ======================
def mahony_ahrs(gyro, accel, q, bias, dt, kp, ki):
    qw, qx, qy, qz = q

    acc_norm = np.linalg.norm(accel)
    if acc_norm < 1e-10:
        q_dot = 0.5 * np.array([
            -qx*gyro[0] - qy*gyro[1] - qz*gyro[2],
             qw*gyro[0] + qy*gyro[2] - qz*gyro[1],
             qw*gyro[1] - qx*gyro[2] + qz*gyro[0],
             qw*gyro[2] + qx*gyro[1] - qy*gyro[0]
        ])
        q_new = q + q_dot * dt
        return q_new / np.linalg.norm(q_new), bias

    acc = accel / acc_norm

    vx = 2 * (qx*qz - qw*qy)
    vy = 2 * (qy*qz + qw*qx)
    vz = 1 - 2 * (qx*qx + qy*qy)

    ex = acc[1]*vz - acc[2]*vy
    ey = acc[2]*vx - acc[0]*vz
    ez = acc[0]*vy - acc[1]*vx

    bias = bias + np.array([ex, ey, ez]) * ki * dt
    gyro_corrected = gyro + np.array([ex, ey, ez]) * kp + bias

    gx, gy, gz = gyro_corrected
    q_dot = 0.5 * np.array([
        -qx*gx - qy*gy - qz*gz,
         qw*gx + qy*gz - qz*gy,
         qw*gy - qx*gz + qz*gx,
         qw*gz + qx*gy - qy*gx
    ])

    q_new = q + q_dot * dt
    return q_new / np.linalg.norm(q_new), bias


# ====================== 核心函数 ======================
def run_mahony(imu_csv, gt_csv, output_dir):
    """
    运行 Mahony 姿态解算
    参数:
        imu_csv:    预处理后的IMU CSV
        gt_csv:     Ground Truth CSV（原始EuRoC格式）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Mahony 姿态解算")
    print("=" * 60)

    # 加载 IMU
    df_imu = pd.read_csv(imu_csv)
    print(f"IMU数据: {len(df_imu)} 行")

    # 加载 GT
    df_gt = None
    has_gt = os.path.exists(gt_csv)
    if has_gt:
        df_gt = pd.read_csv(gt_csv, header=0, names=GT_COLS_RAW)
        df_gt["timestamp_sec"] = df_gt["timestamp_ns"] * 1e-9
        df_gt = df_gt.rename(columns={
            'q_RS_w []': 'qw', 'q_RS_x []': 'qx',
            'q_RS_y []': 'qy', 'q_RS_z []': 'qz'
        })
        print(f"GT数据: {len(df_gt)} 行")
    else:
        print("⚠ 无Ground Truth，仅输出姿态")

    # 运行 Mahony
    n = len(df_imu)
    q = np.array([1.0, 0.0, 0.0, 0.0])
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

    # 与 GT 对比
    rmse_roll = rmse_pitch = rmse_yaw = None
    error_euler = None
    gt_euler = None

    if has_gt:
        gt_time = df_gt["timestamp_sec"].values
        imu_time = np.arange(n) * DT
        gt_indices = np.array([np.argmin(np.abs(gt_time - t)) for t in imu_time])

        gt_euler = np.zeros((n, 3))
        for i, idx in enumerate(gt_indices):
            q_gt = [df_gt["qw"].iloc[idx], df_gt["qx"].iloc[idx],
                    df_gt["qy"].iloc[idx], df_gt["qz"].iloc[idx]]
            gt_euler[i] = Rot.from_quat([q_gt[1], q_gt[2], q_gt[3], q_gt[0]]).as_euler('xyz', degrees=True)

        error_euler = euler_angles - gt_euler
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

    # 可视化
    time = np.arange(n) * DT
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax1 = axes[0, 0]
    ax1.plot(time, euler_angles[:, 0], 'r-', lw=0.8, label='Mahony Roll')
    ax1.plot(time, euler_angles[:, 1], 'g-', lw=0.8, label='Mahony Pitch')
    ax1.plot(time, euler_angles[:, 2], 'b-', lw=0.8, label='Mahony Yaw')
    if has_gt:
        ax1.plot(time, gt_euler[:, 0], 'r--', lw=0.5, alpha=0.5, label='GT Roll')
        ax1.plot(time, gt_euler[:, 1], 'g--', lw=0.5, alpha=0.5, label='GT Pitch')
        ax1.plot(time, gt_euler[:, 2], 'b--', lw=0.5, alpha=0.5, label='GT Yaw')
    ax1.set_xlabel("时间 (s)"); ax1.set_ylabel("角度 (°)")
    ax1.set_title("Mahony 姿态估计"); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    if has_gt:
        ax2 = axes[0, 1]
        ax2.plot(time, error_euler[:, 0], 'r-', lw=0.8, label='Roll 误差')
        ax2.plot(time, error_euler[:, 1], 'g-', lw=0.8, label='Pitch 误差')
        ax2.plot(time, error_euler[:, 2], 'b-', lw=0.8, label='Yaw 误差')
        ax2.axhline(y=0, color='k', ls=':', alpha=0.3)
        ax2.set_xlabel("时间 (s)"); ax2.set_ylabel("误差 (°)")
        ax2.set_title(f"姿态误差 (RMSE: {rmse_roll:.2f}°/{rmse_pitch:.2f}°/{rmse_yaw:.2f}°)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, '无GT数据', transform=axes[0,1].transAxes, ha='center', fontsize=14)

    ax3 = axes[1, 0]
    ax3.plot(time, bias_history[:, 0], 'r-', lw=0.8, label='Bias X')
    ax3.plot(time, bias_history[:, 1], 'g-', lw=0.8, label='Bias Y')
    ax3.plot(time, bias_history[:, 2], 'b-', lw=0.8, label='Bias Z')
    ax3.set_xlabel("时间 (s)"); ax3.set_ylabel("偏置 (rad/s)")
    ax3.set_title("Mahony 陀螺仪偏置估计"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(time, quaternions[:, 0], 'k-', lw=0.8, label='qw')
    ax4.plot(time, quaternions[:, 1], 'r-', lw=0.8, label='qx')
    ax4.plot(time, quaternions[:, 2], 'g-', lw=0.8, label='qy')
    ax4.plot(time, quaternions[:, 3], 'b-', lw=0.8, label='qz')
    ax4.set_xlabel("时间 (s)"); ax4.set_ylabel("四元数值")
    ax4.set_title("Mahony 四元数"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    plt.suptitle(f"Mahony 互补滤波姿态解算 (PI: Kp={KP}, Ki={KI})", fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "mahony_attitude.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

    # 保存
    df_out = pd.DataFrame({
        "time": time,
        "qw": quaternions[:, 0], "qx": quaternions[:, 1],
        "qy": quaternions[:, 2], "qz": quaternions[:, 3],
        "roll": euler_angles[:, 0], "pitch": euler_angles[:, 1], "yaw": euler_angles[:, 2],
        "bias_x": bias_history[:, 0], "bias_y": bias_history[:, 1], "bias_z": bias_history[:, 2]
    })
    out_csv = os.path.join(output_dir, "mahony_attitude.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n结果已保存至: {out_csv}")
    print("Mahony 姿态解算完成！")

    return rmse_roll, rmse_pitch, rmse_yaw


# ====================== 单独运行 ======================
if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_mahony(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        PREPROCESS_DIR = r"D:\同济\euroc_data\数据集\预处理"
        RAW_DATA_DIR = r"D:\同济\euroc_data\数据集"
        MAHONY_DIR = r"D:\同济\euroc_data\数据集\mahony"

        for folder in os.listdir(PREPROCESS_DIR):
            if not os.path.isdir(os.path.join(PREPROCESS_DIR, folder)):
                continue
            imu_csv = os.path.join(PREPROCESS_DIR, folder, f"{folder}_preprocessed.csv")
            gt_csv = os.path.join(RAW_DATA_DIR, folder, "mav0", "state_groundtruth_estimate0", "data.csv")
            output_dir = os.path.join(MAHONY_DIR, folder)

            if os.path.exists(imu_csv):
                print(f"\n处理: {folder}")
                run_mahony(imu_csv, gt_csv, output_dir)
