"""
EuRoC 全序列批量运行脚本
自动遍历所有11个序列，输出每个序列的 RMSE 汇总表
"""

import os
import sys
import yaml
import subprocess
import pandas as pd
import numpy as np
import time

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_single_sequence(seq_name, config):
    """运行单个序列的完整流水线"""
    data_root = config['data_root']
    seq_path = os.path.join(data_root, seq_name)
    output_root = config['output_root']
    seq_output = os.path.join(output_root, seq_name)
    
    os.makedirs(seq_output, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"处理序列: {seq_name}")
    print(f"{'='*60}")
    
    # 检查数据是否存在
    imu_csv = os.path.join(seq_path, "mav0", "imu0", "data.csv")
    gt_csv = os.path.join(seq_path, "mav0", "state_groundtruth_estimate0", "data.csv")
    
    if not os.path.exists(imu_csv):
        print(f"❌ {seq_name}: IMU数据不存在，跳过")
        return None
    if not os.path.exists(gt_csv):
        print(f"❌ {seq_name}: Ground Truth不存在，跳过")
        return None
    
    # 动态导入各个模块
    from preprocessing import run_preprocessing
    from allan_noise import run_allan
    from ekf_fusion import run_ekf
    from mahony_ahrs import run_mahony
    
    # 步骤1: 预处理
    print(f"[1/4] 预处理...")
    preprocessed_csv = os.path.join(seq_output, "preprocessed.csv")
    run_preprocessing(seq_path, preprocessed_csv)
    
    # 步骤2: Allan方差
    print(f"[2/4] Allan方差分析...")
    allan_dir = os.path.join(seq_output, "allan_results")
    noise_json = os.path.join(allan_dir, "imu_noise_params.json")
    run_allan(preprocessed_csv, allan_dir)
    
    # 步骤3: EKF融合
    print(f"[3/4] EKF融合...")
    ekf_dir = os.path.join(seq_output, "ekf_results")
    ekf_csv = os.path.join(ekf_dir, "ekf_trajectory.csv")
    performance_json = os.path.join(ekf_dir, "ekf_performance.json")
    run_ekf(preprocessed_csv, noise_json, gt_csv, ekf_dir)
    
    # 步骤4: Mahony
    print(f"[4/4] Mahony姿态解算...")
    mahony_dir = os.path.join(seq_output, "mahony_results")
    run_mahony(preprocessed_csv, gt_csv, mahony_dir)
    
    # 读取RMSE
    import json
    with open(performance_json, 'r') as f:
        perf = json.load(f)
    
    return {
        "序列": seq_name,
        "RMSE_X (m)": perf["rmse_xyz"][0],
        "RMSE_Y (m)": perf["rmse_xyz"][1],
        "RMSE_Z (m)": perf["rmse_xyz"][2],
        "RMSE_3D (m)": perf["rmse_total"],
        "数据时长 (s)": perf["total_time"]
    }

def main():
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'euroc_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    sequences = config['sequences']
    print(f"共{len(sequences)}个序列待处理:")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {seq}")
    
    # 批量运行
    results = []
    start_time = time.time()
    
    for i, seq in enumerate(sequences, 1):
        print(f"\n{'#'*60}")
        print(f"进度: {i}/{len(sequences)} - {seq}")
        print(f"{'#'*60}")
        
        result = run_single_sequence(seq, config)
        if result:
            results.append(result)
            print(f"✅ {seq}: RMSE_3D = {result['RMSE_3D (m)']:.4f} m")
        else:
            print(f"⏭️ {seq}: 跳过")
    
    # 汇总表
    print(f"\n{'='*80}")
    print("========== EuRoC 全序列 RMSE 汇总 ==========")
    print(f"{'='*80}")
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # 保存汇总表
    summary_csv = os.path.join(config['output_root'], "euroc_all_sequences_rmse.csv")
    df_results.to_csv(summary_csv, index=False, encoding='utf-8')
    print(f"\n汇总表已保存: {summary_csv}")
    
    # 打印统计
    valid_rmse = df_results["RMSE_3D (m)"].dropna()
    if len(valid_rmse) > 0:
        print(f"\n统计信息:")
        print(f"  序列数: {len(valid_rmse)}")
        print(f"  平均RMSE: {valid_rmse.mean():.4f} m")
        print(f"  最小RMSE: {valid_rmse.min():.4f} m")
        print(f"  最大RMSE: {valid_rmse.max():.4f} m")
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    print(f"========== 全部完成 ==========")

if __name__ == "__main__":
    main()
