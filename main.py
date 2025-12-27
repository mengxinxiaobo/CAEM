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
    plt.ylabel('Total Loss (MSE + MMD + Pred)')
    plt.legend()
    plt.grid(True)

    # 保存图片
    plot_path = 'plots/training_loss.png'
    plt.savefig(plot_path)
    print(f"\n[Info] Training plot saved to {plot_path}")
    plt.close()


def main():
    # 0. 初始化环境
    ensure_directories()
    print(f"[Info] Running on device: {tf.config.list_physical_devices('GPU')}")

    # ==========================================
    # 1. 数据准备 (Data Preparation)
    # ==========================================
    print("\n[Step 1] Loading Data...")

    # ---------------------------------------------------------
    # TODO: 在这里替换为你的真实数据读取逻辑
    # 你的数据应该是 (Total_Samples, N, T) 的 numpy 数组
    # ---------------------------------------------------------
    # 模拟数据演示:
    # 假设有 1000 个样本，每个样本包含 27 个传感器，100 个时间步
    # 前 800 个是正常样本 (Label=0)，后 200 个是异常样本 (Label=1)
    TOTAL_SAMPLES = 1000
    mock_data = np.random.randn(TOTAL_SAMPLES, config.NUM_SENSORS, config.TIME_STEPS).astype(np.float32)
    mock_labels = np.array([0] * 800 + [1] * 200)

    # 使用 data_loader 进行处理 (滑动窗口 + 数据切分)
    # X_train 仅包含正常数据
    # X_test 包含混合数据
    X_train, X_val, X_test, y_test = get_dataloaders(mock_data, mock_labels, config)

    print(f"  Training Set: {X_train.shape}")
    print(f"  Validation Set: {X_val.shape}")
    print(f"  Test Set: {X_test.shape}")

    # ==========================================
    # 2. 构建模型 (Model Building)
    # ==========================================
    print("\n[Step 2] Building CAEM Model...")
    model = build_caem_model(config)

    # 编译模型
    # Loss 已经在模型内部通过 add_loss 添加，这里只需指定优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer)

    # 打印模型结构
    # model.summary()

    # ==========================================
    # 3. 定义回调函数 (Callbacks)
    # ==========================================
    callbacks = [
        # A. 只保存验证集 Loss 最小的最佳模型
        ModelCheckpoint(
            filepath='checkpoints/caem_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,  # 仅保存权重，便于加载
            verbose=1
        ),
        # B. 早停：如果 Loss 10 个 epoch 不下降则停止
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # C. 学习率衰减：如果 Loss 不下降，自动减小学习率
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
    # 注意: 因为使用了自定义 Loss Layer，y (目标值) 不需要传递，传 None 即可
    # Keras 会自动调用内部的 loss 计算逻辑
    history = model.fit(
        x=X_train,
        y=None,
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
    # 加载最佳权重 (以防训练最后几个 epoch 过拟合)
    model.load_weights('checkpoints/caem_best_model.h5')

    trainer = CAEMTrainer(model, config)

    # 在(正常的)训练集上计算阈值
    # Logic: Mean(Loss) + Std(Loss)
    trainer.calculate_threshold(X_train)

    # ==========================================
    # 6. 推理与评估 (Evaluation)
    # ==========================================
    print("\n[Step 5] Evaluating on Test Set...")
    results = trainer.evaluate(X_test, y_test)

    print("\n" + "=" * 40)
    print("FINAL EVALUATION RESULTS")
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