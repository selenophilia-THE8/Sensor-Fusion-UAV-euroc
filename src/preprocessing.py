import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import sys

# ====================== 0. 全局常量（所有序列共用） ======================
GRAVITY = 9.81

IMU_COLS_RAW = [
    "timestamp_ns",
    "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
    "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"
]

GT_COLS = [
    "timestamp_ns",
    "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
    "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []",
    "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]",
    "b_w_RS_S_x [rad s^-1]", "b_w_RS_S_y [rad s^-1]", "b_w_RS_S_z [rad s^-1]",
    "b_a_RS_S_x [m s^-2]", "b_a_RS_S_y [m s^-2]", "b_a_RS_S_z [m s^-2]"
]

# ====================== 工具函数（所有序列共用） ======================
def detect_outliers_3sigma(col):
    """检测3σ以外的异常值，返回布尔mask"""
    mean = col.mean()
    std = col.std()
    if std == 0:
        return pd.Series(False, index=col.index)
    lower = mean - 3 * std
    upper = mean + 3 * std
    return (col < lower) | (col > upper)


def replace_outliers_with_linear_interpolation(df, cols):
    """对指定列的异常值用线性插值替换"""
    df_processed = df.copy()
    outlier_stats = {}
    for col in cols:
        outlier_mask = detect_outliers_3sigma(df_processed[col])
        outlier_count = outlier_mask.sum()
        outlier_stats[col] = outlier_count
        if outlier_count > 0:
            df_processed.loc[outlier_mask, col] = np.nan
            df_processed[col] = df_processed[col].interpolate(method='linear')
            print(f"  [{col}] {outlier_count}个异常值已用线性插值替换")
        else:
            print(f"  [{col}] 无异常值")
    return df_processed, outlier_stats


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """四元数转旋转矩阵"""
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),         1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ])
    return R


# ====================== 核心函数：预处理单个序列 ======================
def run_preprocessing(seq_path, output_csv):
    """
    预处理单个 EuRoC 序列
    参数:
        seq_path:   序列根目录，如 D:/同济/euroc_data/数据集/MH_01_easy
        output_csv: 输出CSV路径
    """
    # 自动拼接 IMU 和 GT 路径
    imu_csv = os.path.join(seq_path, "mav0", "imu0", "data.csv")
    gt_csv = os.path.join(seq_path, "mav0", "state_groundtruth_estimate0", "data.csv")

    seq_name = os.path.basename(seq_path)

    # --- 加载 IMU ---
    print("=" * 60)
    print(f"正在加载 {seq_name} IMU 数据...")
    df_imu = pd.read_csv(imu_csv, header=0, names=IMU_COLS_RAW)
    df_imu["timestamp_sec"] = df_imu["timestamp_ns"] * 1e-9
    print(f"  IMU 数据大小: {df_imu.shape}")
    print(f"  时间范围: {df_imu['timestamp_sec'].min():.2f}s ~ {df_imu['timestamp_sec'].max():.2f}s")

    # --- 加载 GT ---
    print(f"正在加载 {seq_name} Ground Truth 数据...")
    df_gt = pd.read_csv(gt_csv, header=0, names=GT_COLS)
    df_gt["timestamp_sec"] = df_gt["timestamp_ns"] * 1e-9
    print(f"  GT 数据大小: {df_gt.shape}")

    # --- 重命名列 ---
    df_imu = df_imu.rename(columns={
        "w_RS_S_x [rad s^-1]": "wx_raw",
        "w_RS_S_y [rad s^-1]": "wy_raw",
        "w_RS_S_z [rad s^-1]": "wz_raw",
        "a_RS_S_x [m s^-2]": "ax_raw",
        "a_RS_S_y [m s^-2]": "ay_raw",
        "a_RS_S_z [m s^-2]": "az_raw"
    })
    IMU_COLS = ["wx_raw", "wy_raw", "wz_raw", "ax_raw", "ay_raw", "az_raw"]

    # --- 缺失值检测 ---
    print("\n=== 缺失值检测 ===")
    missing_info = df_imu[IMU_COLS].isnull().sum()
    if missing_info.sum() > 0:
        print(missing_info[missing_info > 0])
        df_imu[IMU_COLS] = df_imu[IMU_COLS].fillna(method='ffill')
        print("  已用前向填充处理缺失值")
    else:
        print("  ✓ 无缺失值")

    # --- 异常值处理 ---
    print("\n=== 异常值处理 ===")
    df_imu, outlier_stats = replace_outliers_with_linear_interpolation(df_imu, IMU_COLS)

    # --- 时间基准统一 ---
    t0 = df_imu["timestamp_sec"].iloc[0]
    df_imu["relative_time"] = df_imu["timestamp_sec"] - t0
    df_gt["relative_time"] = df_gt["timestamp_sec"] - t0
    print(f"\n=== 时间基准统一 ===")
    print(f"  IMU时间范围(相对): 0 ~ {df_imu['relative_time'].iloc[-1]:.2f} 秒")

    # --- 三轴重力补偿 ---
    print("\n=== 重力补偿（三轴姿态） ===")
    n_imu = len(df_imu)
    gravity_compensated = np.zeros((n_imu, 3))

    gt_time = df_gt["relative_time"].values
    gt_qw = df_gt["q_RS_w []"].values
    gt_qx = df_gt["q_RS_x []"].values
    gt_qy = df_gt["q_RS_y []"].values
    gt_qz = df_gt["q_RS_z []"].values

    g_world = np.array([0, 0, GRAVITY])

    for i in range(n_imu):
        t_imu = df_imu["relative_time"].iloc[i]
        idx = np.argmin(np.abs(gt_time - t_imu))
        R_wb = quaternion_to_rotation_matrix(gt_qw[idx], gt_qx[idx], gt_qy[idx], gt_qz[idx])
        g_body = R_wb.T @ g_world
        a_raw = np.array([df_imu["ax_raw"].iloc[i],
                          df_imu["ay_raw"].iloc[i],
                          df_imu["az_raw"].iloc[i]])
        gravity_compensated[i, :] = a_raw - g_body

    df_imu["ax_compensated"] = gravity_compensated[:, 0]
    df_imu["ay_compensated"] = gravity_compensated[:, 1]
    df_imu["az_compensated"] = gravity_compensated[:, 2]
    print("  重力补偿完成")

    # --- 数据标准化 ---
    print("\n=== 数据标准化 ===")
    scaler = StandardScaler()
    std_cols = ["wx_raw", "wy_raw", "wz_raw", "ax_compensated", "ay_compensated", "az_compensated"]
    df_imu[["wx_std", "wy_std", "wz_std", "ax_std", "ay_std", "az_std"]] = scaler.fit_transform(df_imu[std_cols])
    print("  标准化完成")

    # --- 保存 ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_imu.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n=== 预处理结果已保存 ===")
    print(f"  保存路径: {output_csv}")
    print(f"  数据行数: {len(df_imu)}")

    # --- 可视化 ---
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    time_imu = df_imu["relative_time"]
    # 子图1
    ax1 = axes[0, 0]
    ax1.plot(time_imu, df_imu["az_raw"], label="原始Z轴加速度", alpha=0.7, linewidth=0.5)
    ax1.plot(time_imu, df_imu["az_compensated"], label="补偿后Z轴加速度", alpha=0.7, linewidth=0.5)
    ax1.set_xlabel("相对时间（秒）"); ax1.set_ylabel("加速度（m/s²）")
    ax1.set_title(f"重力补偿对比（Z轴，{seq_name}）"); ax1.legend(); ax1.grid(True, alpha=0.3)

    # 子图2
    ax2 = axes[0, 1]
    for col, label in zip(["wx_std", "wy_std", "wz_std"], ["Wx", "Wy", "Wz"]):
        ax2.plot(time_imu, df_imu[col], label=label, alpha=0.7, linewidth=0.5)
    ax2.set_xlabel("相对时间（秒）"); ax2.set_ylabel("标准化值")
    ax2.set_title("标准化后角速度"); ax2.legend(); ax2.grid(True, alpha=0.3)

    # 子图3
    ax3 = axes[1, 0]
    for col, label in zip(["ax_std", "ay_std", "az_std"], ["Ax", "Ay", "Az"]):
        ax3.plot(time_imu, df_imu[col], label=label, alpha=0.7, linewidth=0.5)
    ax3.set_xlabel("相对时间（秒）"); ax3.set_ylabel("标准化值")
    ax3.set_title("标准化后加速度（三轴补偿后）"); ax3.legend(); ax3.grid(True, alpha=0.3)

    # 子图4
    ax4 = axes[1, 1]
    ax4.plot(df_gt["p_RS_R_x [m]"], df_gt["p_RS_R_y [m]"], linewidth=0.8, color='blue')
    ax4.set_xlabel("X位置（米）"); ax4.set_ylabel("Y位置（米）")
    ax4.set_title(f"Ground Truth 2D轨迹 ({seq_name})"); ax4.grid(True, alpha=0.3); ax4.axis('equal')

    plt.suptitle(f"EuRoC {seq_name} 数据预处理结果", fontsize=14)
    plt.tight_layout()
    # 保存图片到输出目录
    plot_path = output_csv.replace(".csv", "_preprocessing.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"  可视化已保存: {plot_path}")
    print("=" * 60)

    return df_imu, df_gt  # 返回数据框，方便后续模块调用


# ====================== 单独运行 ======================
if __name__ == "__main__":
    if len(sys.argv) == 3:
        # 命令行模式: python preprocessing.py <seq_path> <output_csv>
        run_preprocessing(sys.argv[1], sys.argv[2])
    else:
        # 默认模式：自动遍历预处理目录下所有序列
        DATA_ROOT = r"D:\同济\euroc_data\数据集"
        OUTPUT_DIR = r"D:\同济\euroc_data\数据集\预处理"
        
        # 自动扫描 DATA_ROOT 下所有包含 mav0 的文件夹（即 EuRoC 序列）
        for folder in os.listdir(DATA_ROOT):
            seq_path = os.path.join(DATA_ROOT, folder)
            imu_csv = os.path.join(seq_path, "mav0", "imu0", "data.csv")
            if os.path.exists(imu_csv):
                output_csv = os.path.join(OUTPUT_DIR, folder, f"{folder}_preprocessed.csv")
                run_preprocessing(seq_path, output_csv)
