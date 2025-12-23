import os
import numpy as np
import tensorflow as tf
from src import config
from src.caem import build_caem_model
from src.data_loader import get_dataloaders
from src.trainer import CAEMTrainer

# 设置随机种子以复现
tf.random.set_seed(42)
np.random.seed(42)


def main():
    # 1. 准备数据
    # 这里我们使用随机数据模拟你的输入，请替换为真实数据加载逻辑
    print("Generating Mock Data (Replace with real data loading)...")
    TOTAL_SAMPLES = 1000
    # 模拟数据: (1000, 27, 100)
    mock_data = np.random.randn(TOTAL_SAMPLES, config.NUM_SENSORS, config.TIME_STEPS).astype(np.float32)
    # 模拟标签: 前800正常(0), 后200异常(1)
    mock_labels = np.array([0] * 800 + [1] * 200)

    # 获取处理好的数据
    X_train, X_val, X_test, y_test = get_dataloaders(mock_data, mock_labels, config)

    # 2. 构建模型
    print("Building CAEM Model...")
    model = build_caem_model(config)

    # 编译 (Loss已在内部通过 add_loss 添加，这里只需指定优化器)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE))
    model.summary()

    # 3. 训练模型 (Algorithm 1: Training)
    print("Starting Training...")
    # 由于 Loss 是自包含的，y_true 可以为 None (Keras要求有占位符，但不会被使用)
    # 但 model.fit 通常需要 y，我们可以传 dummy y 或者使用 Dataset
    # 这里的 outputs 有5个，所以我们不传 y，让 Keras 的 add_loss 自己工作
    model.fit(
        X_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, None)
    )

    # 4. 阈值计算与评估 (Algorithm 1: Inference)
    trainer = CAEMTrainer(model, config)

    # 在训练集上计算阈值
    trainer.calculate_threshold(X_train)

    # 在测试集上评估
    results = trainer.evaluate(X_test, y_test)

    print("\n=== Evaluation Results ===")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("Confusion Matrix:")
    print(results['confusion_matrix'])


if __name__ == "__main__":
    main()