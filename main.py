# import os
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#
# # 导入自定义模块
# from src import config
# from src.caem import build_caem_model
# from src.data_loader import get_dataloaders
# from src.trainer import CAEMTrainer
#
# # 1. 设置随机种子 (确保实验可复现)
# tf.random.set_seed(42)
# np.random.seed(42)
#
#
# def ensure_directories():
#     """创建必要的文件夹"""
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
#     if not os.path.exists('logs'):
#         os.makedirs('logs')
#     if not os.path.exists('plots'):
#         os.makedirs('plots')
#
#
# def plot_training_history(history):
#     """绘制并保存 Loss 曲线"""
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, loss, 'b-', label='Training Loss')
#     plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
#     plt.title('CAEM Training Convergence')
#     plt.xlabel('Epochs')
#     plt.ylabel('Total Loss')
#     plt.legend()
#     plt.grid(True)
#
#     # 保存图片
#     plot_path = 'plots/training_loss.png'
#     plt.savefig(plot_path)
#     print(f"\n[Info] Training plot saved to {plot_path}")
#     plt.close()
#
#
# def segment_time_series(data, labels, window_size, step):
#     """
#     辅助函数: 将连续的时间序列流切分成 (样本数, 传感器数, 时间步) 的格式
#     Args:
#         data: (Total_Time, N_Sensors) e.g., (1500000, 27)
#         labels: (Total_Time, )
#         window_size: T (时间步长, e.g., 100)
#         step: 滑动步长 (e.g., 100 为不重叠, 50 为半重叠)
#     Returns:
#         segments: (N_Samples, N_Sensors, T)
#         seg_labels: (N_Samples, )
#     """
#     segments = []
#     seg_labels = []
#
#     # 遍历数据，按 step 步长滑动
#     # 比如总长 100万，每次取 100 个点
#     for i in range(0, len(data) - window_size, step):
#         # 提取窗口数据 (T, N)
#         window_data = data[i: i + window_size]
#         window_label = labels[i: i + window_size]
#
#         # 维度变换: (T, N) -> (N, T) 以匹配模型输入要求
#         # 例如 (100, 27) -> (27, 100)
#         segments.append(window_data.T)
#
#         # 标签策略: 如果窗口内超过一半是异常(1)，则该样本标记为异常
#         if np.sum(window_label) > (window_size // 2):
#             seg_labels.append(1)
#         else:
#             seg_labels.append(0)
#
#     return np.array(segments), np.array(seg_labels)
#
#
# def main():
#     # 0. 初始化环境
#     ensure_directories()
#     print(f"[Info] Running on device: {tf.config.list_physical_devices('GPU')}")
#
#     # ==========================================
#     # 1. 数据准备 (Data Preparation)
#     # ==========================================
#     print("\n[Step 1] Loading Real PAMAP2 Data...")
#
#     # 加载预处理好的 .npy 文件
#     data_path = 'data/processed/pamap2_caem.npy'
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Data not found at {data_path}. Please run src/preprocess_pamap2.py first!")
#
#     loaded = np.load(data_path, allow_pickle=True).item()
#     raw_stream_data = loaded['data']  # Shape: (Total_Time, 27)
#     raw_stream_labels = loaded['label']  # Shape: (Total_Time, )
#
#     print(f"  Raw Stream Data Shape: {raw_stream_data.shape}")
#
#     # --- 切片处理 (Segmentation) ---
#     # 使用非重叠切片 (step=100) 快速生成样本
#     # config.TIME_STEPS 通常为 100
#     WINDOW_SIZE = config.TIME_STEPS
#     STEP = 100
#
#     print("  Segmenting time series (this may take a moment)...")
#     X_segmented, y_segmented = segment_time_series(
#         raw_stream_data,
#         raw_stream_labels,
#         WINDOW_SIZE,
#         STEP
#     )
#
#     # 维度确认: (样本数, 27, 100)
#     print(f"  Segmented Data Shape: {X_segmented.shape}")
#
#     # 传入 get_dataloaders 进行 训练/验证/测试 划分 (5:1:4)
#     # 这里会自动生成 Sequence Window (Batch, h, 27, 100, 1)
#     X_train, X_val, X_test, y_test = get_dataloaders(X_segmented, y_segmented, config)
#
#     print(f"  Training Set (Sequence): {X_train.shape}")
#     print(f"  Validation Set (Sequence): {X_val.shape}")
#     print(f"  Test Set (Sequence): {X_test.shape}")
#
#     # ==========================================
#     # 2. 构建模型 (Model Building)
#     # ==========================================
#     print("\n[Step 2] Building CAEM Model...")
#     model = build_caem_model(config)
#
#     # 编译模型
#     # Loss 已在模型内部通过 add_loss 添加，这里只需指定优化器
#     optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
#     model.compile(optimizer=optimizer)
#
#     # ==========================================
#     # 3. 定义回调函数 (Callbacks)
#     # ==========================================
#     callbacks = [
#         # A. 只保存验证集 Loss 最小的最佳模型
#         ModelCheckpoint(
#             filepath='checkpoints/caem_best_model.h5',
#             monitor='val_loss',
#             save_best_only=True,
#             save_weights_only=True,
#             verbose=1
#         ),
#         # B. 早停：如果 Loss 10 个 epoch 不下降则停止
#         EarlyStopping(
#             monitor='val_loss',
#             patience=10,
#             restore_best_weights=True,
#             verbose=1
#         ),
#         # C. 学习率衰减
#         ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=5,
#             min_lr=1e-6,
#             verbose=1
#         )
#     ]
#
#     # ==========================================
#     # 4. 模型训练 (Training)
#     # ==========================================
#     print("\n[Step 3] Starting Training...")
#     # 注意: 因为数据量变大了，如果是 CPU 跑，建议先去 config.py 把 EPOCHS 改小一点测试一下
#     history = model.fit(
#         x=X_train,
#         y=None,  # 自定义 Loss 层不需要 y
#         epochs=config.EPOCHS,
#         batch_size=config.BATCH_SIZE,
#         validation_data=(X_val, None),
#         callbacks=callbacks,
#         shuffle=True
#     )
#
#     # 绘制训练曲线
#     plot_training_history(history)
#
#     # ==========================================
#     # 5. 阈值计算 (Thresholding)
#     # ==========================================
#     print("\n[Step 4] Calculating Anomaly Threshold...")
#     # 加载最佳权重
#     model.load_weights('checkpoints/caem_best_model.h5')
#
#     trainer = CAEMTrainer(model, config)
#
#     # 在正常的训练集上计算阈值 (Mean + Std)
#     trainer.calculate_threshold(X_train)
#
#     # ==========================================
#     # 6. 推理与评估 (Evaluation)
#     # ==========================================
#     print("\n[Step 5] Evaluating on Test Set...")
#     results = trainer.evaluate(X_test, y_test)
#
#     print("\n" + "=" * 40)
#     print("FINAL EVALUATION RESULTS (PAMAP2)")
#     print("=" * 40)
#     print(f"Precision: {results['precision']:.4f}")
#     print(f"Recall:    {results['recall']:.4f}")
#     print(f"F1 Score:  {results['f1']:.4f}")
#     print("-" * 20)
#     print("Confusion Matrix:")
#     print(results['confusion_matrix'])
#     print("=" * 40)
#
#
# if __name__ == "__main__":
#     main()



###############################################################################################


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 导入自定义模块
from src import config
# 注意: 这里导入的是 GCN 版本的模型构建函数
from src.caem import build_caem_gcn_model
from src.data_loader import get_dataloaders
from src.trainer import CAEMTrainer
import matplotlib.pyplot as plt

# --- 解决中文显示问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 1. 设置随机种子 (确保实验结果可以复现)
tf.random.set_seed(42)
np.random.seed(42)


def ensure_directories():
    """创建必要的文件夹 (如果不存在的话)"""
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('plots'):
        os.makedirs('plots')


def plot_training_history(history):
    """
    绘制并保存训练过程中的 Loss 曲线
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='训练集 Loss')
    plt.plot(epochs, val_loss, 'r-', label='验证集 Loss')
    plt.title('GCN-CAEM 训练收敛曲线')
    plt.xlabel('Epochs (轮数)')
    plt.ylabel('Total Loss (总损失)')
    plt.legend()
    plt.grid(True)

    # 保存图片
    plot_path = 'plots/training_loss_gcn.png'
    plt.savefig(plot_path)
    print(f"\n[提示] 训练曲线图已保存至: {plot_path}")
    plt.close()


def segment_time_series(data, labels, window_size, step):
    """
    辅助函数: 将连续的时间序列流切分成 (样本数, 传感器数, 时间步) 的格式

    参数:
        data: 原始数据流 (Total_Time, N_Sensors)
        labels: 原始标签流 (Total_Time, )
        window_size: 窗口大小 (即时间步 T, 例如 100)
        step: 滑动步长 (例如 100 表示不重叠切片)
    """
    segments = []
    seg_labels = []

    # 遍历数据，按 step 步长滑动窗口
    for i in range(0, len(data) - window_size, step):
        window_data = data[i: i + window_size]
        window_label = labels[i: i + window_size]

        # 维度变换: 从 (T, N) 变为 (N, T) 以适配模型输入
        segments.append(window_data.T)

        # 标签策略: 如果窗口内超过一半的数据是异常(1)，则该样本标记为异常
        if np.sum(window_label) > (window_size // 2):
            seg_labels.append(1)
        else:
            seg_labels.append(0)

    return np.array(segments), np.array(seg_labels)


def get_adjacency_matrix(data, threshold=0.7):
    """
    [改进点 1]：从二值图升级为加权图
    """
    print(f"\n[图构建] 正在计算加权邻接矩阵 (阈值: {threshold})...")
    N = data.shape[-2]
    sample_data = data[:2000]
    reshaped = np.transpose(sample_data, (0, 1, 3, 2)).reshape(-1, N)
    corr_matrix = np.corrcoef(reshaped, rowvar=False)

    # 核心修改：保留相关系数的绝对值作为权重，而不是直接变成 1.0
    # 低于阈值的切断（置0），高于阈值的保留原始相关度
    adj = np.where(np.abs(corr_matrix) >= threshold, np.abs(corr_matrix), 0.0)

    # 加上自环（节点自身的权重）
    np.fill_diagonal(adj, 1.0)

    # 对称归一化 (D^-0.5 * A * D^-0.5)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    return norm_adj.astype(np.float32)


def main():
    # 0. 初始化环境
    ensure_directories()
    print(f"[系统] 运行设备: {tf.config.list_physical_devices('GPU')}")

    # ==========================================
    # 1. 数据准备
    # ==========================================
    print("\n[步骤 1] 加载真实 PAMAP2 数据...")

    # 加载预处理好的 .npy 文件
    data_path = 'data/processed/pamap2_caem.npy'
    # 提示: 如果你要跑 DSADS 数据集，请去 config.py 修改 NUM_SENSORS 并在这里更改路径
    # data_path = 'data/processed/dsads_caem.npy'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    loaded = np.load(data_path, allow_pickle=True).item()
    raw_stream_data = loaded['data']
    raw_stream_labels = loaded['label']

    # --- 切片处理 ---
    WINDOW_SIZE = config.TIME_STEPS
    STEP = 100  # 不重叠切片

    print("  正在进行时间序列切片...")
    X_segmented, y_segmented = segment_time_series(
        raw_stream_data,
        raw_stream_labels,
        WINDOW_SIZE,
        STEP
    )

    # 获取数据加载器 (默认带有 Channel 维度: Batch, H, N, T, 1)
    X_train, X_val, X_test, y_test = get_dataloaders(X_segmented, y_segmented, config)

    # !!! 关键修改 !!!
    # GCN 不需要最后的 Channel 维度 (..., 1)
    # 我们把它去掉: (Batch, H, N, T, 1) -> (Batch, H, N, T)
    print("  [GCN 适配] 正在移除通道维度 (Squeezing)...")
    X_train = np.squeeze(X_train, axis=-1)
    X_val = np.squeeze(X_val, axis=-1)
    X_test = np.squeeze(X_test, axis=-1)

    print(f"  训练集形状 (GCN 就绪): {X_train.shape}")
    print(f"  验证集形状: {X_val.shape}")

    # ==========================================
    # 2. 构建图与模型
    # ==========================================

    # A. 计算邻接矩阵
    # 使用训练集数据自动构建传感器关系图
    adj_matrix = get_adjacency_matrix(X_train, threshold=0.7)

    print("\n[步骤 2] 构建 GCN-CAEM 模型...")

    # B. 传入邻接矩阵构建模型
    model = build_caem_gcn_model(config, adj_matrix)

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer)

    # 如果想查看模型结构，可以取消下面这行的注释
    # model.summary()

    # ==========================================
    # 3. 训练准备
    # ==========================================
    callbacks = [
        # 保存最佳模型权重
        ModelCheckpoint(
            filepath='checkpoints/gcn_caem_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        # 早停机制: 10轮不提升则停止
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减: 5轮不提升则减半
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ==========================================
    # 4. 开始训练
    # ==========================================
    print("\n[步骤 3] 开始训练...")
    history = model.fit(
        x=X_train,
        y=None,  # 无监督学习，无需标签
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, None),
        callbacks=callbacks,
        shuffle=True
    )

    # 绘制曲线
    plot_training_history(history)

    # ==========================================
    # 5. 阈值计算
    # ==========================================
    print("\n[步骤 4] 计算异常检测阈值...")
    # 加载刚才训练最好的权重
    model.load_weights('checkpoints/gcn_caem_best_model.h5')

    trainer = CAEMTrainer(model, config)

    # 在训练集上计算正常样本的重构误差分布，从而确定阈值
    trainer.calculate_threshold(X_train)

    # 提示: 如果发现误报太多(Precision低)，可以在这里手动提高阈值
    # trainer.threshold = trainer.threshold_mean + 2.0 * trainer.threshold_std

    # ==========================================
    # 6. 最终评估
    # ==========================================
    print("\n[步骤 5] 在测试集上评估性能...")
    results = trainer.evaluate(X_test, y_test)

    print("\n" + "=" * 40)
    print("最终评估结果 (GCN-PAMAP2)")
    print("=" * 40)
    print(f"精确率 (Precision): {results['precision']:.4f}")
    print(f"召回率 (Recall):    {results['recall']:.4f}")
    print(f"F1 分数 (F1 Score): {results['f1']:.4f}")
    print("-" * 20)
    print("混淆矩阵 (Confusion Matrix):")
    print(results['confusion_matrix'])
    print("=" * 40)


if __name__ == "__main__":
    main()