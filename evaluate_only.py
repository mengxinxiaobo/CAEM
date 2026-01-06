import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ================================================================
# 解决库冲突，确保在 CPU/GPU 上平稳运行
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ================================================================

# 导入自定义模块
from src import config
from src.caem import build_caem_gcn_model  # 确保此函数已包含残差和Bi-LSTM逻辑
from src.data_loader import get_dataloaders
from src.trainer import CAEMTrainer

# 1. 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)


def segment_time_series(data, labels, window_size, step):
    """辅助函数: 切分时间序列"""
    segments = []
    seg_labels = []
    for i in range(0, len(data) - window_size, step):
        window_data = data[i: i + window_size]
        window_label = labels[i: i + window_size]
        segments.append(window_data.T)
        if np.sum(window_label) > (window_size // 2):
            seg_labels.append(1)
        else:
            seg_labels.append(0)
    return np.array(segments), np.array(seg_labels)


def get_adjacency_matrix(data, threshold=0.7):
    """
    【同步修改】：计算加权邻接矩阵
    必须与训练时的加权逻辑完全一致，否则权重加载后特征会错位。
    """
    print(f"\n[Graph] Re-building Weighted Adjacency Matrix (threshold={threshold})...")
    N = data.shape[-2]
    # 使用训练集的前2000个样本计算传感器相关性
    sample_data = data[:2000]
    reshaped = np.transpose(sample_data, (0, 1, 3, 2)).reshape(-1, N)
    corr_matrix = np.corrcoef(reshaped, rowvar=False)

    # 保留相关系数绝对值作为权重
    adj = np.where(np.abs(corr_matrix) >= threshold, np.abs(corr_matrix), 0.0)
    np.fill_diagonal(adj, 1.0)

    # 对称归一化: D^-0.5 * A * D^-0.5
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    return norm_adj.astype(np.float32)


def main():
    print(f"[Info] Evaluation Mode Running on: {tf.config.list_physical_devices('GPU')}")

    # ==========================================
    # 1. 数据准备
    # ==========================================
    print("\n[Step 1] Loading Data...")
    data_path = 'data/processed/pamap2_caem.npy'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run preprocessing first.")
        return

    loaded = np.load(data_path, allow_pickle=True).item()
    raw_stream_data = loaded['data']
    raw_stream_labels = loaded['label']

    X_segmented, y_segmented = segment_time_series(
        raw_stream_data, raw_stream_labels, config.TIME_STEPS, 100
    )

    X_train, X_val, X_test, y_test = get_dataloaders(X_segmented, y_segmented, config)

    # GCN 维度适配 (移除通道维度)
    X_train = np.squeeze(X_train, axis=-1)
    X_test = np.squeeze(X_test, axis=-1)

    print(f"  Training Set Shape: {X_train.shape}")
    print(f"  Test Set Shape: {X_test.shape}")

    # ==========================================
    # 2. 构建模型并加载权重
    # ==========================================
    # 重要：这里的阈值建议设为 0.7，与您最新的训练代码保持同步
    adj_matrix = get_adjacency_matrix(X_train, threshold=0.7)

    print("\n[Step 2] Building Model (Residual GCN + Bi-LSTM) & Loading Weights...")
    # 这里的 build_caem_gcn_model 内部必须包含 residual_gcn_block 逻辑
    model = build_caem_gcn_model(config, adj_matrix)

    checkpoint_path = 'checkpoints/gcn_caem_best_model.h5'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}，请先运行训练脚本！")

    model.load_weights(checkpoint_path)
    print("  SUCCESS: Weights loaded successfully!")

    # ==========================================
    # 3. 评估执行
    # ==========================================
    print("\n[Step 3] Recalculating Threshold & Evaluating...")

    # 实例化 Trainer。注意：此时它会加载 src/trainer.py 中您最新修改的
    # 异常分数计算公式 (例如 lambda_mse=0.3, lambda_pred=1.7)
    trainer = CAEMTrainer(model, config)

    # 1. 重新计算阈值。
    # 建议手动在 src/trainer.py 调整 k=0.5 之后再运行此脚本
    trainer.calculate_threshold(X_train)

    # 2. 在测试集执行推理
    results = trainer.evaluate(X_test, y_test)

    print("\n" + "=" * 50)
    print("   GCN-CAEM FINAL EVALUATION (Weighted + Bi-LSTM)")
    print("=" * 50)
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    print("=" * 50)


if __name__ == "__main__":
    main()