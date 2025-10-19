#!/usr/bin/env python3
"""
分析所有50个胚胎的特征
找出模式、异常值、统计信息
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def analyze_all_embryos():
    print("="*60)
    print("📊 分析所有胚胎特征")
    print("="*60)
    
    # 收集所有胚胎的特征
    embryo_data = []
    
    for z_file in sorted(Path("latents_unique").glob("*_z.npy")):
        embryo_id = z_file.stem.replace("_z", "")
        z = np.load(z_file)  # [16, 128]
        
        # 计算各种特征
        # 1. 速度
        speeds = np.linalg.norm(z[1:] - z[:-1], axis=1)  # [15]
        mean_speed = speeds.mean()
        std_speed = speeds.std()
        max_speed = speeds.max()
        
        # 2. 轨迹长度
        traj_length = speeds.sum()
        
        # 3. 起点-终点距离
        start_end_dist = np.linalg.norm(z[0] - z[-1])
        
        # 4. 特征统计
        mean_feature = z.mean()
        std_feature = z.std()
        
        embryo_data.append({
            'embryo_id': embryo_id,
            'mean_speed': mean_speed,
            'std_speed': std_speed,
            'max_speed': max_speed,
            'traj_length': traj_length,
            'start_end_dist': start_end_dist,
            'mean_feature': mean_feature,
            'std_feature': std_feature
        })
    
    # 转成 DataFrame
    df = pd.DataFrame(embryo_data)
    
    print(f"\n✅ 分析了 {len(df)} 个胚胎")
    print(f"\n📊 统计摘要:")
    print(df[['mean_speed', 'traj_length', 'start_end_dist']].describe())
    
    # 保存结果
    df.to_csv("embryo_features_summary.csv", index=False)
    print(f"\n✅ 特征摘要已保存: embryo_features_summary.csv")
    
    # 找出异常值
    print(f"\n{'='*60}")
    print(f"🔍 异常检测")
    print(f"{'='*60}")
    
    # 速度异常高的
    speed_threshold = df['mean_speed'].mean() + 2 * df['mean_speed'].std()
    speed_outliers = df[df['mean_speed'] > speed_threshold]
    print(f"\n⚠️  发育速度异常快的胚胎 (>{speed_threshold:.4f}):")
    for _, row in speed_outliers.iterrows():
        print(f"   {row['embryo_id']}: 速度 {row['mean_speed']:.4f}")
    
    # 速度异常低的
    speed_low_threshold = df['mean_speed'].mean() - 2 * df['mean_speed'].std()
    speed_low_outliers = df[df['mean_speed'] < speed_low_threshold]
    print(f"\n⚠️  发育速度异常慢的胚胎 (<{speed_low_threshold:.4f}):")
    for _, row in speed_low_outliers.iterrows():
        print(f"   {row['embryo_id']}: 速度 {row['mean_speed']:.4f}")
    
    # 轨迹长度异常的
    traj_threshold = df['traj_length'].mean() + 2 * df['traj_length'].std()
    traj_outliers = df[df['traj_length'] > traj_threshold]
    print(f"\n⚠️  发育轨迹异常长的胚胎 (>{traj_threshold:.4f}):")
    for _, row in traj_outliers.iterrows():
        print(f"   {row['embryo_id']}: 长度 {row['traj_length']:.4f}")
    
    # 可视化分布
    print(f"\n{'='*60}")
    print(f"📈 生成分布图")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Speed Distribution
    axes[0,0].hist(df['mean_speed'], bins=20, edgecolor='black', alpha=0.7)
    axes[0,0].axvline(df['mean_speed'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,0].set_title('Development Speed Distribution', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Mean Speed', fontsize=10)
    axes[0,0].set_ylabel('Count', fontsize=10)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Trajectory Length Distribution
    axes[0,1].hist(df['traj_length'], bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0,1].axvline(df['traj_length'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,1].set_title('Trajectory Length Distribution', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Trajectory Length', fontsize=10)
    axes[0,1].set_ylabel('Count', fontsize=10)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Speed Variability Distribution
    axes[1,0].hist(df['std_speed'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1,0].axvline(df['std_speed'].mean(), color='red', linestyle='--', label='Mean')
    axes[1,0].set_title('Speed Variability Distribution', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Speed Std Dev', fontsize=10)
    axes[1,0].set_ylabel('Count', fontsize=10)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Speed vs Trajectory Length Scatter
    axes[1,1].scatter(df['mean_speed'], df['traj_length'], alpha=0.6, label='Normal')
    axes[1,1].set_title('Speed vs Trajectory Length', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Mean Speed', fontsize=10)
    axes[1,1].set_ylabel('Trajectory Length', fontsize=10)
    axes[1,1].grid(True, alpha=0.3)
    
    # Mark outliers
    for _, row in speed_outliers.iterrows():
        axes[1,1].scatter(row['mean_speed'], row['traj_length'], 
                         color='red', s=150, marker='x', linewidths=3)
    
    # Add outlier label only once
    if len(speed_outliers) > 0:
        axes[1,1].scatter([], [], color='red', s=150, marker='x', 
                         linewidths=3, label='Abnormal (Fast)')
    
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig("all_embryos_analysis.png", dpi=150)
    print(f"✅ 分布图已保存: all_embryos_analysis.png")
    
    # 排名
    print(f"\n{'='*60}")
    print(f"🏆 胚胎排名")
    print(f"{'='*60}")
    
    print(f"\n发育速度最快的前5个:")
    top_speed = df.nlargest(5, 'mean_speed')
    for i, (_, row) in enumerate(top_speed.iterrows(), 1):
        print(f"  {i}. {row['embryo_id']}: {row['mean_speed']:.4f}")
    
    print(f"\n发育速度最慢的前5个:")
    bottom_speed = df.nsmallest(5, 'mean_speed')
    for i, (_, row) in enumerate(bottom_speed.iterrows(), 1):
        print(f"  {i}. {row['embryo_id']}: {row['mean_speed']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"✅ 分析完成！")
    print(f"{'='*60}")
    print(f"\n📁 产生的文件:")
    print(f"  - embryo_features_summary.csv (所有特征数据)")
    print(f"  - all_embryos_analysis.png (分布图)")
    print(f"\n📊 查看方式:")
    print(f"  cat embryo_features_summary.csv | head")
    print(f"  在 Cursor 中打开 all_embryos_analysis.png")
    
    return df

if __name__ == "__main__":
    df = analyze_all_embryos()

