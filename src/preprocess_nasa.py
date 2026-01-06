import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 这里填写您的 "archive" 文件夹的绝对路径
# 注意：根据您的 tree 结果，数据在 data/data/train 下
BASE_DIR = r"D:\Study\pycharm-projects\CAEM\data\archive"

# 2. 关键子目录 (根据您的 tree 结果调整)
TRAIN_DIR = os.path.join(BASE_DIR, "data", "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "data", "test")
LABEL_FILE = os.path.join(BASE_DIR, "labeled_anomalies.csv")

# 3. 选择数据集 ('MSL' 或 'SMAP')
DATASET_NAME = 'MSL'

# 4. 输出路径
OUTPUT_FILE = f"data/processed/nasa_{DATASET_NAME.lower()}_caem.npy"


# ===========================================

def load_and_merge_data():
    print(f"🚀 正在处理 NASA {DATASET_NAME} 数据集...")
    print(f"   源目录: {TRAIN_DIR}")

    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError(f"找不到标签文件: {LABEL_FILE}")

    # 1. 读取标签文件，筛选属于当前数据集的通道
    label_df = pd.read_csv(LABEL_FILE)
    target_df = label_df[label_df['spacecraft'] == DATASET_NAME]

    # 获取属于 MSL 的所有文件名 (如 M-1, T-2 等)
    chan_ids = target_df['chan_id'].values
    print(f"   找到 {len(chan_ids)} 个子数据集 (Entities).")

    all_train_data = []
    all_test_data = []
    all_test_labels = []

    # 2. 遍历合并
    # 策略：CAEM 需要单一的大矩阵。我们将所有子数据集在“时间”维度上拼接。
    # 虽然物理上 M-1 和 M-2 是不同的部件，但为了训练一个通用的 GCN，我们将它们视为连续的数据流。

    valid_channels = 0
    for chan in tqdm(chan_ids):
        train_path = os.path.join(TRAIN_DIR, f"{chan}.npy")
        test_path = os.path.join(TEST_DIR, f"{chan}.npy")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            continue

        # 读取数据 (Time, Feats)
        # MSL 的 Feats 通常是 55, SMAP 是 25
        t_train = np.load(train_path)
        t_test = np.load(test_path)

        # 确保维度一致
        if len(t_train.shape) < 2: t_train = t_train.reshape(-1, 1)
        if len(t_test.shape) < 2: t_test = t_test.reshape(-1, 1)

        # 读取该通道的异常标签
        row = target_df[target_df['chan_id'] == chan].iloc[0]
        anom_seqs = ast.literal_eval(row['anomaly_sequences'])  # "[[10, 20], [50, 60]]"

        # 生成 0/1 标签向量
        label_arr = np.zeros(len(t_test), dtype=int)
        for seq in anom_seqs:
            start, end = seq
            # 修正索引越界
            start = max(0, start)
            end = min(end, len(t_test))
            label_arr[start:end] = 1

        # 添加到列表
        all_train_data.append(t_train)
        all_test_data.append(t_test)
        all_test_labels.append(label_arr)
        valid_channels += 1

    print(f"   成功加载 {valid_channels} 个有效通道。正在拼接...")

    # 3. 拼接数据
    # 最终形状: (Total_Time, 55) for MSL
    X_train = np.concatenate(all_train_data, axis=0)
    X_test = np.concatenate(all_test_data, axis=0)
    y_test = np.concatenate(all_test_labels, axis=0)

    # 训练集标签设为全 0
    y_train = np.zeros(len(X_train), dtype=int)

    # 4. 归一化 (Z-Score)
    print("   正在归一化...")
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    # 避免除以 0
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # 5. 合并保存
    # 格式对齐 PAMAP2: 一个大字典，包含 data 和 label
    final_data = np.concatenate([X_train, X_test], axis=0)
    final_label = np.concatenate([y_train, y_test], axis=0)

    # 创建目录并保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.save(OUTPUT_FILE, {'data': final_data, 'label': final_label})

    print(f"\n✅ 处理完成!")
    print(f"   输出文件: {OUTPUT_FILE}")
    print(f"   数据形状 (Time, Nodes): {final_data.shape}")
    print(f"   MSL 特征数应为 55, SMAP 应为 25. 当前为: {final_data.shape[1]}")


if __name__ == "__main__":
    load_and_merge_data()