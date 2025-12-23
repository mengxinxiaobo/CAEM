import numpy as np
import tensorflow as tf


def create_sliding_window(data, window_size):
    """
    将 (Samples, N, T, 1) 转换为 (Samples-h+1, h, N, T, 1)
    """
    num_samples = len(data)
    X = []
    # 比如: 0,1,2,3,4 -> 预测 5 (如果h=5, 我们这里生成包含当前时刻的序列)
    # Memory Network 需要输入前 h-1 个，预测第 h 个
    # 所以我们的窗口长度是 h
    for i in range(num_samples - window_size + 1):
        window = data[i: i + window_size]
        X.append(window)
    return np.array(X)


def get_dataloaders(raw_data, raw_labels, config, normal_class=0):
    """
    数据分割与加载
    raw_data: (Total, N, T) -> 需要增加 Channel 维度
    """
    # 1. 增加 Channel 维度 -> (Total, N, T, 1)
    if raw_data.ndim == 3:
        raw_data = np.expand_dims(raw_data, axis=-1)

    print(f"Raw Data Shape: {raw_data.shape}")

    # 2. 分离正常和异常
    normal_data = raw_data[raw_labels == normal_class]
    abnormal_data = raw_data[raw_labels != normal_class]

    # 3. 划分 Train/Val/Test (5:1:4) for Normal
    n_normal = len(normal_data)
    n_train = int(n_normal * 0.5)
    n_val = int(n_normal * 0.1)

    # 这里的切分必须在滑动窗口 *之前* 还是 *之后* ?
    # 论文Impl Details: "split normal samples... into...".
    # 通常是先切分数据集，再各自做滑动窗口，以防止数据泄漏。

    train_normal = normal_data[:n_train]
    val_normal = normal_data[n_train: n_train + n_val]
    test_normal = normal_data[n_train + n_val:]

    # 4. 生成滑动窗口 (Sequence Generation)
    # 只有生成序列后，才能 Shuffle
    X_train = create_sliding_window(train_normal, config.MEMORY_WINDOW)
    X_val = create_sliding_window(val_normal, config.MEMORY_WINDOW)

    # 测试集: 正常 + 异常
    X_test_normal = create_sliding_window(test_normal, config.MEMORY_WINDOW)
    X_test_abnormal = create_sliding_window(abnormal_data, config.MEMORY_WINDOW)

    X_test = np.concatenate([X_test_normal, X_test_abnormal], axis=0)

    # 生成测试标签 (用于评估)
    # 0 for normal, 1 for abnormal
    y_test = np.concatenate([
        np.zeros(len(X_test_normal)),
        np.ones(len(X_test_abnormal))
    ], axis=0)

    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape: {X_test.shape}")

    return X_train, X_val, X_test, y_test