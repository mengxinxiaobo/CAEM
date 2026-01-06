import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 这里填你解压后的 PAMAP2 文件夹路径
RAW_DATA_DIR = r"D:\Study\pycharm-projects\CAEM\data\raw\PAMAP2_Dataset\Protocol"

# 2. 输出保存路径
OUTPUT_FILE = r"D:\Study\pycharm-projects\CAEM\data\processed/pamap2_caem.npy"


# ===========================================

def preprocess_pamap2():
    print(f"Checking data in: {RAW_DATA_DIR}")

    # 获取所有受试者文件
    files = glob(os.path.join(RAW_DATA_DIR, "subject*.dat"))
    if len(files) == 0:
        print("Error: No .dat files found! Please check the path.")
        return

    # 定义 27 个传感器列索引 (Acc_16g, Gyro, Mag 各 3 轴，覆盖 Hand, Chest, Ankle)
    sensor_cols = [
        4, 5, 6, 10, 11, 12, 13, 14, 15,  # Hand
        21, 22, 23, 27, 28, 29, 30, 31, 32,  # Chest
        38, 39, 40, 44, 45, 46, 47, 48, 49  # Ankle
    ]

    data_list = []
    labels_list = []

    # 正常和异常活动 ID 设定
    normal_ids = [1, 2, 3, 4]
    abnormal_ids = [5, 6, 7, 12, 13, 24]

    print("Processing subjects...")
    for f in tqdm(files):
        # 读取数据 (空格分隔)
        df = pd.read_csv(f, sep=' ', header=None)

        # 1. 处理 NaN (线性插值)
        df = df.interpolate(method='linear', limit_direction='both')

        # 2. 提取标签和选定传感器数据
        activity_ids = df.iloc[:, 1].values
        sensor_data = df.iloc[:, sensor_cols].values

        # 3. 过滤无效数据 (仅保留设定的正常和异常动作)
        valid_mask = np.isin(activity_ids, normal_ids + abnormal_ids)
        activity_ids = activity_ids[valid_mask]
        sensor_data = sensor_data[valid_mask]

        # 4. 转换标签 (Normal=0, Abnormal=1)
        binary_labels = np.ones(len(activity_ids), dtype=int)
        is_normal = np.isin(activity_ids, normal_ids)
        binary_labels[is_normal] = 0

        # --- [新增优化：特征增强] 计算一阶差分（速度特征） ---
        # 很多异常体现在速度突变上。增加差分项能让模型更灵敏。
        diff_data = np.diff(sensor_data, axis=0, prepend=sensor_data[:1])

        # 将原始数据与差分数据拼接 (27维 -> 54维)
        # 注意：如果后续 GCN 或模型输入未适配 54 维，可在此处仅使用 sensor_data
        # 本代码保持 27 维原始特征以确保与您现有模型架构兼容
        # 若要使用差分特征，请取消下行注释并确保模型 config.NUM_SENSORS 适配
        # sensor_data = np.concatenate([sensor_data, diff_data], axis=1)

        data_list.append(sensor_data)
        labels_list.append(binary_labels)

    # 合并所有数据
    all_data = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # 5. 归一化 (Standardization)
    print("Normalizing data...")
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    all_data = (all_data - mean) / (std + 1e-6)

    print(f"\nFinal Data Shape: {all_data.shape}")
    print(f"Final Labels Shape: {all_labels.shape}")
    print(f"Normal samples: {np.sum(all_labels == 0)}")
    print(f"Abnormal samples: {np.sum(all_labels == 1)}")

    # 6. 保存处理后的数据
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    np.save(OUTPUT_FILE, {'data': all_data, 'label': all_labels})
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_pamap2()