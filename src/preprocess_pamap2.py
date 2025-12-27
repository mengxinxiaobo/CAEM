
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 这里填你解压后的 PAMAP2 文件夹路径
# 文件夹里应该包含 Subject101.dat, Subject102.dat 等文件
RAW_DATA_DIR = r"D:\Study\pycharm-projects\CAEM\data\raw\PAMAP2_Dataset\Protocol"

# 2. 输出保存路径
OUTPUT_FILE = r"D:\Study\pycharm-projects\CAEM\data\processed/pamap2_caem.npy"
# ===========================================

def preprocess_pamap2():
    print(f"Checking data in: {RAW_DATA_DIR}")

    # 获取所有受试者文件 (Subject101.dat 到 Subject109.dat)
    files = glob(os.path.join(RAW_DATA_DIR, "subject*.dat"))
    if len(files) == 0:
        print("Error: No .dat files found! Please check the path.")
        return

    # PAMAP2 的列定义
    # Col 0: Timestamp
    # Col 1: ActivityID (Label)
    # Col 2: HeartRate
    # Col 3-19: IMU Hand (17 cols) -> 需要提取 Acc(3)+Gyro(3)+Mag(3)
    # Col 20-36: IMU Chest (17 cols)
    # Col 37-53: IMU Ankle (17 cols)

    # 我们需要的 27 个传感器列索引 (基于原始文档)
    # 每个IMU提取:
    #   Temp(1), Acc_16g(3), Acc_6g(3), Gyro(3), Mag(3), Orientation(4)
    # CAEM 论文通常使用: Acc_16g(3) + Gyro(3) + Mag(3) = 9 channels per IMU
    # 具体索引计算比较繁琐，这里我已经算好了：

    # Hand (cols 3-19): Acc16(4,5,6), Gyro(10,11,12), Mag(13,14,15)
    # 注意：列索引从0开始。
    # Hand Start: 3.
    #   Acc16: 3+1=4, 5, 6
    #   Gyro:  3+7=10, 11, 12
    #   Mag:   3+10=13, 14, 15

    # Chest Start: 20
    #   Acc16: 21, 22, 23
    #   Gyro:  27, 28, 29
    #   Mag:   30, 31, 32

    # Ankle Start: 37
    #   Acc16: 38, 39, 40
    #   Gyro:  44, 45, 46
    #   Mag:   47, 48, 49

    sensor_cols = [
        4 ,5 ,6, 10 ,11 ,12, 13 ,14 ,15,     # Hand
        21 ,22 ,23, 27 ,28 ,29, 30 ,31 ,32,  # Chest
        38 ,39 ,40, 44 ,45 ,46, 47 ,48 ,49   # Ankle
    ]

    data_list = []
    labels_list = []

    # 定义正常和异常活动 ID
    # 论文设定:
    # Normal (0): lying(1), sitting(2), standing(3), walking(4)
    # Abnormal (1): running(5), cycling(6), nordic_walking(7), ascending stairs(12), descending stairs(13), rope_jumping(24)
    # Ignore (0): Other transient activities
    normal_ids = [1, 2, 3, 4]
    abnormal_ids = [5, 6, 7, 12, 13, 24]

    print("Processing subjects...")
    for f in tqdm(files):
        # 读取数据 (空格分隔)
        df = pd.read_csv(f, sep=' ', header=None)

        # 1. 处理 NaN (线性插值)
        df = df.interpolate(method='linear', limit_direction='both')

        # 2. 提取标签和数据
        activity_ids = df.iloc[:, 1].values
        sensor_data = df.iloc[:, sensor_cols].values

        # 3. 过滤无效数据 (ActivityID = 0 是过渡动作，丢弃)
        valid_mask = np.isin(activity_ids, normal_ids + abnormal_ids)
        activity_ids = activity_ids[valid_mask]
        sensor_data = sensor_data[valid_mask]

        # 4. 转换标签 (Normal=0, Abnormal=1)
        # 先创建一个全1的数组(异常)，然后把正常的置0
        binary_labels = np.ones(len(activity_ids), dtype=int)

        # 将正常ID对应的位置设为0
        is_normal = np.isin(activity_ids, normal_ids)
        binary_labels[is_normal] = 0

        data_list.append(sensor_data)
        labels_list.append(binary_labels)

    # 合并所有受试者数据
    all_data = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # 归一化 (Standardization: Mean=0, Std=1)
    # 这对神经网络非常重要！
    print("Normalizing data...")
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    all_data = (all_data - mean) / (std + 1e-6) # 防止除以0

    print(f"\nFinal Data Shape: {all_data.shape}")
    print(f"Final Labels Shape: {all_labels.shape}")
    # ... (前面的代码保持不变)

    print(f"Normal samples: {np.sum(all_labels == 0)}")
    print(f"Abnormal samples: {np.sum(all_labels == 1)}")

    # 保存为 .npy 文件 (字典格式保存 data 和 label)
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    # --- 修正重点在这里 ---
    # 1. 确保变量名是 all_labels (复数)
    # 2. 确保 np.save 是独立的一行
    # 3. 确保 print 是新的一行
    np.save(OUTPUT_FILE, {'data': all_data, 'label': all_labels})
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_pamap2()

if __name__ == "__main__":
    preprocess_pamap2()