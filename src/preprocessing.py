import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# ====================== 1. 基础配置（EuRoC 数据集） ======================
# EuRoC 数据路径（修改为你的实际路径）
DATASET_PATH = r"D:\同济\euroc_data\数据集\MH_01_easy"  # 示例：MH_01序列

# IMU 数据文件（EuRoC 标准结构）
IMU_CSV = os.path.join(DATASET_PATH, "mav0", "imu0", "data.csv")

# Ground Truth 数据文件（用于对比验证）
GT_CSV = os.path.join(DATASET_PATH, "mav0", "state_groundtruth_estimate0", "data.csv")

# 重力常量
GRAVITY = 9.81

# EuRoC IMU 数据列名（注意：第一列是时间戳纳秒）
IMU_COLS_RAW = [
    "timestamp_ns",           # 时间戳（纳秒）
    "w_RS_S_x [rad s^-1]",   # 角速度 x
    "w_RS_S_y [rad s^-1]",   # 角速度 y
    "w_RS_S_z [rad s^-1]",   # 角速度 z
    "a_RS_S_x [m s^-2]",     # 加速度 x
    "a_RS_S_y [m s^-2]",     # 加速度 y
    "a_RS_S_z [m s^-2]"      # 加速度 z
]

# Ground Truth 列名
GT_COLS = [
    "timestamp_ns",
    "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
    "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []",
    "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]",
    "b_w_RS_S_x [rad s^-1]", "b_w_RS_S_y [rad s^-1]", "b_w_RS_S_z [rad s^-1]",
    "b_a_RS_S_x [m s^-2]", "b_a_RS_S_y [m s^-2]", "b_a_RS_S_z [m s^-2]"
]

# ====================== 2. 加载 IMU 数据 ======================
print("=" * 60)
print("正在加载 EuRoC IMU 数据...")
df_imu = pd.read_csv(IMU_CSV, header=0, names=IMU_COLS_RAW)

# 转换时间戳：纳秒 → 秒
df_imu["timestamp_sec"] = df_imu["timestamp_ns"] * 1e-9

print(f"IMU 数据大小: {df_imu.shape}")
print(f"IMU 列名: {df_imu.columns.tolist()}")
print(f"时间范围: {df_imu['timestamp_sec'].min():.2f}s ~ {df_imu['timestamp_sec'].max():.2f}s")

# ====================== 3. 加载 Ground Truth 数据 ======================
print("\n正在加载 Ground Truth 数据...")
df_gt = pd.read_csv(GT_CSV, header=0, names=GT_COLS)

# 转换时间戳
df_gt["timestamp_sec"] = df_gt["timestamp_ns"] * 1e-9

print(f"Ground Truth 数据大小: {df_gt.shape}")
print(f"时间范围: {df_gt['timestamp_sec'].min():.2f}s ~ {df_gt['timestamp_sec'].max():.2f}s")

# ====================== 4. 将 IMU 列重命名为简洁名称 ======================
df_imu = df_imu.rename(columns={
    "w_RS_S_x [rad s^-1]": "wx_raw",
    "w_RS_S_y [rad s^-1]": "wy_raw",
    "w_RS_S_z [rad s^-1]": "wz_raw",
    "a_RS_S_x [m s^-2]": "ax_raw",
    "a_RS_S_y [m s^-2]": "ay_raw",
    "a_RS_S_z [m s^-2]": "az_raw"
})

IMU_COLS = ["wx_raw", "wy_raw", "wz_raw", "ax_raw", "ay_raw", "az_raw"]

# ====================== 5. 缺失值检测 ======================
missing_info = df_imu[IMU_COLS].isnull().sum()
print("\n=== 缺失值检测 ===")
if missing_info.sum() > 0:
    print(missing_info[missing_info > 0])
    # 对IMU数据用前向填充（时间序列特性）
    df_imu[IMU_COLS] = df_imu[IMU_COLS].fillna(method='ffill')
    print("已用前向填充处理缺失值")
else:
    print("✓ 无缺失值")

# ====================== 6. 异常值检测（3σ准则）与线性插值 ======================
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
    """
    对指定列的异常值用线性插值替换
    """
    df_processed = df.copy()
    outlier_stats = {}

    for col in cols:
        outlier_mask = detect_outliers_3sigma(df_processed[col])
        outlier_count = outlier_mask.sum()
        outlier_stats[col] = outlier_count

        if outlier_count > 0:
            df_processed.loc[outlier_mask, col] = np.nan
            df_processed[col] = df_processed[col].interpolate(method='linear')
            print(f"[{col}] {outlier_count}个异常值已用线性插值替换")
        else:
            print(f"[{col}] 无异常值，无需处理")

    return df_processed, outlier_stats

print("\n=== 异常值处理 ===")
df_imu, outlier_stats = replace_outliers_with_linear_interpolation(df_imu, IMU_COLS)

# ====================== 7. 时间基准统一 ======================
t0 = df_imu["timestamp_sec"].iloc[0]
df_imu["relative_time"] = df_imu["timestamp_sec"] - t0
df_gt["relative_time"] = df_gt["timestamp_sec"] - t0

print(f"\n=== 时间基准统一 ===")
print(f"IMU时间范围(相对): 0 ~ {df_imu['relative_time'].iloc[-1]:.2f} 秒")
print(f"GT 时间范围(相对): {df_gt['relative_time'].min():.2f} ~ {df_gt['relative_time'].max():.2f} 秒")

# ====================== 8. 三轴重力补偿（从Ground Truth获取姿态） ======================
print("\n=== 重力补偿（基于真实姿态，不再是简单的Z轴减去9.81） ===")

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """四元数转旋转矩阵"""
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),         1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ])
    return R

# 对齐：找到每个IMU时间戳最接近的Ground Truth姿态
print("正在进行IMU与Ground Truth时间对齐...")
n_imu = len(df_imu)
gravity_compensated = np.zeros((n_imu, 3))  # ax, ay, az补偿后

# Ground Truth数据转为numpy，加速计算
gt_time = df_gt["relative_time"].values
gt_qw = df_gt["q_RS_w []"].values
gt_qx = df_gt["q_RS_x []"].values
gt_qy = df_gt["q_RS_y []"].values
gt_qz = df_gt["q_RS_z []"].values

for i in range(n_imu):
    t_imu = df_imu["relative_time"].iloc[i]
    # 找最接近的Ground Truth索引
    idx = np.argmin(np.abs(gt_time - t_imu))
    
    # 获取旋转矩阵
    R_wb = quaternion_to_rotation_matrix(gt_qw[idx], gt_qx[idx], gt_qy[idx], gt_qz[idx])
    
    # 重力在世界坐标系是 [0, 0, -9.81]
    g_world = np.array([0, 0, GRAVITY])  # 向下为正（与数据集定义有关）
    
    # 将重力旋转到机体坐标系，然后补偿
    g_body = R_wb.T @ g_world  # 世界→机体
    
    # 原始加速度
    a_raw = np.array([
        df_imu["ax_raw"].iloc[i],
        df_imu["ay_raw"].iloc[i],
        df_imu["az_raw"].iloc[i]
    ])
    
    # 补偿后的加速度（减去机体坐标系下感受到的重力）
    a_compensated = a_raw - g_body
    
    gravity_compensated[i, :] = a_compensated

# 添加补偿后的列
df_imu["ax_compensated"] = gravity_compensated[:, 0]
df_imu["ay_compensated"] = gravity_compensated[:, 1]
df_imu["az_compensated"] = gravity_compensated[:, 2]

print("重力补偿完成（三轴姿态补偿）")
print("补偿后加速度预览（前3行）：")
print(df_imu[["ax_raw", "ax_compensated", "ay_raw", "ay_compensated", "az_raw", "az_compensated"]].head(3))

# ====================== 9. 数据标准化 ======================
print("\n=== 数据标准化 ===")
scaler = StandardScaler()
std_cols = ["wx_raw", "wy_raw", "wz_raw", "ax_compensated", "ay_compensated", "az_compensated"]
df_imu[["wx_std", "wy_std", "wz_std", "ax_std", "ay_std", "az_std"]] = scaler.fit_transform(df_imu[std_cols])

print("标准化后数据预览（前3行）：")
print(df_imu[["wx_std", "wy_std", "wz_std", "ax_std", "ay_std", "az_std"]].head(3))

# ====================== 10. 保存预处理结果 ======================
OUTPUT_CSV = r"D:\同济\euroc_data\数据集\预处理.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_imu.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\n=== 预处理结果已保存 ===")
print(f"保存路径: {OUTPUT_CSV}")

# ====================== 11. 预处理效果可视化 ======================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：重力补偿对比（Z轴）
ax1 = axes[0, 0]
time_imu = df_imu["relative_time"]
ax1.plot(time_imu, df_imu["az_raw"], label="原始Z轴加速度", alpha=0.7, linewidth=0.5)
ax1.plot(time_imu, df_imu["az_compensated"], label="补偿后Z轴加速度（三轴姿态）", alpha=0.7, linewidth=0.5)
ax1.set_xlabel("相对时间（秒）")
ax1.set_ylabel("加速度（m/s²）")
ax1.set_title("重力补偿对比（Z轴，EuRoC MH_01）")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：标准化后角速度
ax2 = axes[0, 1]
ax2.plot(time_imu, df_imu["wx_std"], label="Wx 标准化", alpha=0.7, linewidth=0.5)
ax2.plot(time_imu, df_imu["wy_std"], label="Wy 标准化", alpha=0.7, linewidth=0.5)
ax2.plot(time_imu, df_imu["wz_std"], label="Wz 标准化", alpha=0.7, linewidth=0.5)
ax2.set_xlabel("相对时间（秒）")
ax2.set_ylabel("标准化值")
ax2.set_title("标准化后角速度")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：标准化后加速度（补偿后）
ax3 = axes[1, 0]
ax3.plot(time_imu, df_imu["ax_std"], label="Ax 标准化", alpha=0.7, linewidth=0.5)
ax3.plot(time_imu, df_imu["ay_std"], label="Ay 标准化", alpha=0.7, linewidth=0.5)
ax3.plot(time_imu, df_imu["az_std"], label="Az 标准化", alpha=0.7, linewidth=0.5)
ax3.set_xlabel("相对时间（秒）")
ax3.set_ylabel("标准化值")
ax3.set_title("标准化后加速度（三轴补偿后）")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4：Ground Truth 3D轨迹
ax4 = axes[1, 1]
ax4.plot(df_gt["p_RS_R_x [m]"], df_gt["p_RS_R_y [m]"], linewidth=0.8, color='blue')
ax4.set_xlabel("X位置（米）")
ax4.set_ylabel("Y位置（米）")
ax4.set_title("Ground Truth 2D轨迹")
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

plt.suptitle(f"EuRoC MH_01 数据预处理结果\n代码已从悬停IMU适配为EuRoC数据集", fontsize=14)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("=== 预处理流程全部完成 ===")
print(f"预处理后数据行数: {len(df_imu)}")
print(f"输出文件: {OUTPUT_CSV}")
print("=" * 60)
