import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ================================================================
# 环境设置
# ================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置随机种子，保证复现性
tf.random.set_seed(42)
np.random.seed(42)

# 导入自定义模块
from src import config
from src.caem import build_caem_gcn_model
from src.data_loader import get_dataloaders
from src.trainer import CAEMTrainer


def segment_time_series(data, labels, window_size, step):
    """
    滑动窗口切分时间序列
    :param step: 步长。
                 Training时建议设小(如 50)以增加数据量(Overlapping)；
                 Testing时建议设为 window_size 以避免测试集泄漏。
    """
    segments = []
    seg_labels = []
    # 确保不越界
    for i in range(0, len(data) - window_size, step):
        window_data = data[i: i + window_size]
        window_label = labels[i: i + window_size]

        # 转置为 (N, T) 以适配 GCN 输入习惯，后续在 loader 里可能调整
        segments.append(window_data.T)

        # 标签策略：窗口内超过一半异常则标记为异常
        if np.sum(window_label) > (window_size // 2):
            seg_labels.append(1)
        else:
            seg_labels.append(0)

    return np.array(segments), np.array(seg_labels)


def get_weighted_adj(data, threshold=0.7):
    """
    构建加权邻接矩阵
    :param data: 训练数据 (Samples, N, T)
    :param threshold: 皮尔逊相关系数阈值
    """
    print(f"\n[Graph] 正在构建图结构 (Corr Threshold={threshold})...")
    N = data.shape[-2]  # 节点数

    # 取一部分数据计算相关性，避免内存爆炸
    sample_size = min(len(data), 2000)
    sample_data = data[:sample_size]

    # 变形为 (Samples * T, N) 进行相关性计算
    reshaped = np.transpose(sample_data, (0, 1, 3, 2)).reshape(-1, N)

    # 计算绝对值相关系数
    corr_matrix = np.abs(np.corrcoef(reshaped, rowvar=False))

    # 阈值过滤
    adj = np.where(corr_matrix >= threshold, corr_matrix, 0.0)

    # 自连接 (对角线置 1)
    np.fill_diagonal(adj, 1.0)

    # 对称归一化 (D^-0.5 * A * D^-0.5)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = np.diag(d_inv_sqrt)

    norm_adj = adj.dot(d_mat).transpose().dot(d_mat)
    return norm_adj.astype(np.float32)


def main():
    print(f"[Info] 当前数据集: {config.DATASET_NAME}")
    print(f"[Info] 传感器维度: {config.NUM_SENSORS}")
    print(f"[Info] 数据路径: {config.DATA_PATH}")
    print(f"[Info] 运行设备: {tf.config.list_physical_devices('GPU')}")

    # ==========================================
    # 1. 数据加载与预处理
    # ==========================================
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"数据文件不存在: {config.DATA_PATH}\n请先运行对应数据集的 preprocess 脚本。")

    print("\n[Step 1] Loading Data...")
    loaded = np.load(config.DATA_PATH, allow_pickle=True).item()
    data_all = loaded['data']
    label_all = loaded['label']

    # 简单划分 Train/Test (按 80% 截断，模拟时间流)
    # 注意：更严谨的做法是在 preprocess 里分好，这里简化处理
    split_idx = int(len(data_all) * 0.8)
    train_raw = data_all[:split_idx]
    train_label = label_all[:split_idx]
    test_raw = data_all[split_idx:]
    test_label = label_all[split_idx:]

    print(f"   Train Raw Shape: {train_raw.shape}")
    print(f"   Test Raw Shape:  {test_raw.shape}")

    # ==========================================
    # 2. 滑动窗口切分
    # ==========================================
    print("\n[Step 2] Segmenting Time Series...")

    # 关键策略：训练集使用高重叠 (Step < Window) 以增加样本量
    # NASA数据较少，建议 step=50 (重叠50%)
    TRAIN_STEP = 50
    TEST_STEP = config.TIME_STEPS  # 测试集不重叠

    X_train_seg, y_train_seg = segment_time_series(train_raw, train_label, config.TIME_STEPS, TRAIN_STEP)
    X_test_seg, y_test_seg = segment_time_series(test_raw, test_label, config.TIME_STEPS, TEST_STEP)

    print(f"   Train Segments: {X_train_seg.shape}")

    # 获取 DataLoader (划分 Val)
    # 注意：get_dataloaders 会处理维度扩展 (Batch, N, T, 1)
    X_train, X_val, X_test, y_test = get_dataloaders(X_train_seg, y_train_seg, config)

    # 修正维度适配 GCN: (Batch, N, T, 1) -> (Batch, N, T)
    # 因为 GCN 层通常接受 (Batch, N, T) 或 (Batch, N, Features)
    X_train = np.squeeze(X_train, axis=-1)
    X_val = np.squeeze(X_val, axis=-1)
    # 测试集这里先不 squeeze，因为 evaluate 里可能还需要处理，或者保持统一
    # 为了保险，我们在这里统一 squeeze
    X_test_input = np.squeeze(X_test, axis=-1)

    # ==========================================
    # 3. 构建模型
    # ==========================================
    print("\n[Step 3] Building Model...")

    # 构建邻接矩阵
    adj_matrix = get_weighted_adj(X_train, threshold=0.6)  # NASA数据可以尝试稍微稀疏一点的图

    model = build_caem_gcn_model(config, adj_matrix)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer)

    # model.summary() # 可选：打印模型结构

    # ==========================================
    # 4. 训练配置
    # ==========================================
    # 自动根据数据集名称保存到不同文件
    best_model_path = os.path.join(config.CHECKPOINT_DIR, f'gcn_caem_best_model.h5')

    callbacks = [
        ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True, monitor='val_loss'),
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss')
    ]

    print(f"\n[Step 4] Starting Training for {config.EPOCHS} epochs...")
    print(f"   Checkpoints will be saved to: {best_model_path}")

    history = model.fit(
        X_train, y=None,  # 无监督/自监督，Target由模型内部生成
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, None),
        callbacks=callbacks,
        shuffle=True
    )

    # ==========================================
    # 5. 初始评估 (Sanity Check)
    # ==========================================
    print("\n[Step 5] Calculating Initial Threshold...")

    # 实例化 Trainer
    trainer = CAEMTrainer(model, config)

    # 计算阈值
    trainer.calculate_threshold(X_train)

    # 简单评估
    print("Running initial evaluation on Test set...")
    # 注意：这里需要传入真实的 Test Segments 和 Labels
    # 由于 X_test 是从 get_dataloaders 出来的，可能经过了 shuffle，
    # 这里的 y_test 对应的是 get_dataloaders 返回的 y_test

    results = trainer.evaluate(X_test_input, y_test)

    print(f"\n>>> Training Complete. Best F1 (Initial): {results['f1']:.4f}")
    print("请使用 'evaluate_only.py' 进行详细调优和 PA 指标计算。")


if __name__ == "__main__":
    main()