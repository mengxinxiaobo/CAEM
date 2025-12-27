import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 导入自定义模块
from src import config
from src.caem import build_caem_model
from src.data_loader import get_dataloaders
from src.trainer import CAEMTrainer

# 1. 设置随机种子 (确保实验可复现)
tf.random.set_seed(42)
np.random.seed(42)


def ensure_directories():
    """创建必要的文件夹"""
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('plots'):
        os.makedirs('plots')


def plot_training_history(history):
    """绘制并保存 Loss 曲线"""
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('CAEM Training Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.grid(True)

    # 保存图片
    plot_path = 'plots/training_loss.png'
    plt.savefig(plot_path)
    print(f"\n[Info] Training plot saved to {plot_path}")
    plt.close()


def segment_time_series(data, labels, window_size, step):
    """
    辅助函数: 将连续的时间序列流切分成 (样本数, 传感器数, 时间步) 的格式
    Args:
        data: (Total_Time, N_Sensors) e.g., (1500000, 27)
        labels: (Total_Time, )
        window_size: T (时间步长, e.g., 100)
        step: 滑动步长 (e.g., 100 为不重叠, 50 为半重叠)
    Returns:
        segments: (N_Samples, N_Sensors, T)
        seg_labels: (N_Samples, )
    """
    segments = []
    seg_labels = []

    # 遍历数据，按 step 步长滑动
    # 比如总长 100万，每次取 100 个点
    for i in range(0, len(data) - window_size, step):
        # 提取窗口数据 (T, N)
        window_data = data[i: i + window_size]
        window_label = labels[i: i + window_size]

        # 维度变换: (T, N) -> (N, T) 以匹配模型输入要求
        # 例如 (100, 27) -> (27, 100)
        segments.append(window_data.T)

        # 标签策略: 如果窗口内超过一半是异常(1)，则该样本标记为异常
        if np.sum(window_label) > (window_size // 2):
            seg_labels.append(1)
        else:
            seg_labels.append(0)

    return np.array(segments), np.array(seg_labels)


def main():
    # 0. 初始化环境
    ensure_directories()
    print(f"[Info] Running on device: {tf.config.list_physical_devices('GPU')}")

    # ==========================================
    # 1. 数据准备 (Data Preparation)
    # ==========================================
    print("\n[Step 1] Loading Real PAMAP2 Data...")

    # 加载预处理好的 .npy 文件
    data_path = 'data/processed/pamap2_caem.npy'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Please run src/preprocess_pamap2.py first!")

    loaded = np.load(data_path, allow_pickle=True).item()
    raw_stream_data = loaded['data']  # Shape: (Total_Time, 27)
    raw_stream_labels = loaded['label']  # Shape: (Total_Time, )

    print(f"  Raw Stream Data Shape: {raw_stream_data.shape}")

    # --- 切片处理 (Segmentation) ---
    # 使用非重叠切片 (step=100) 快速生成样本
    # config.TIME_STEPS 通常为 100
    WINDOW_SIZE = config.TIME_STEPS
    STEP = 100

    print("  Segmenting time series (this may take a moment)...")
    X_segmented, y_segmented = segment_time_series(
        raw_stream_data,
        raw_stream_labels,
        WINDOW_SIZE,
        STEP
    )

    # 维度确认: (样本数, 27, 100)
    print(f"  Segmented Data Shape: {X_segmented.shape}")

    # 传入 get_dataloaders 进行 训练/验证/测试 划分 (5:1:4)
    # 这里会自动生成 Sequence Window (Batch, h, 27, 100, 1)
    X_train, X_val, X_test, y_test = get_dataloaders(X_segmented, y_segmented, config)

    print(f"  Training Set (Sequence): {X_train.shape}")
    print(f"  Validation Set (Sequence): {X_val.shape}")
    print(f"  Test Set (Sequence): {X_test.shape}")

    # ==========================================
    # 2. 构建模型 (Model Building)
    # ==========================================
    print("\n[Step 2] Building CAEM Model...")
    model = build_caem_model(config)

    # 编译模型
    # Loss 已在模型内部通过 add_loss 添加，这里只需指定优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer)

    # ==========================================
    # 3. 定义回调函数 (Callbacks)
    # ==========================================
    callbacks = [
        # A. 只保存验证集 Loss 最小的最佳模型
        ModelCheckpoint(
            filepath='checkpoints/caem_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        # B. 早停：如果 Loss 10 个 epoch 不下降则停止
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # C. 学习率衰减
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ==========================================
    # 4. 模型训练 (Training)
    # ==========================================
    print("\n[Step 3] Starting Training...")
    # 注意: 因为数据量变大了，如果是 CPU 跑，建议先去 config.py 把 EPOCHS 改小一点测试一下
    history = model.fit(
        x=X_train,
        y=None,  # 自定义 Loss 层不需要 y
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, None),
        callbacks=callbacks,
        shuffle=True
    )

    # 绘制训练曲线
    plot_training_history(history)

    # ==========================================
    # 5. 阈值计算 (Thresholding)
    # ==========================================
    print("\n[Step 4] Calculating Anomaly Threshold...")
    # 加载最佳权重
    model.load_weights('checkpoints/caem_best_model.h5')

    trainer = CAEMTrainer(model, config)

    # 在正常的训练集上计算阈值 (Mean + Std)
    trainer.calculate_threshold(X_train)

    # ==========================================
    # 6. 推理与评估 (Evaluation)
    # ==========================================
    print("\n[Step 5] Evaluating on Test Set...")
    results = trainer.evaluate(X_test, y_test)

    print("\n" + "=" * 40)
    print("FINAL EVALUATION RESULTS (PAMAP2)")
    print("=" * 40)
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("-" * 20)
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    print("=" * 40)


if __name__ == "__main__":
    main()