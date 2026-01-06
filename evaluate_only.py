import os
import numpy as np
import tensorflow as tf
import csv
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ================================================================
# 解决库冲突，确保在 CPU/GPU 上平稳运行
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ================================================================

# 导入自定义模块
from src import config
from src.caem import build_caem_gcn_model
from src.data_loader import get_dataloaders
from src.trainer import CAEMTrainer

# 1. 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)


def apply_point_adjustment(y_true, y_pred):
    """
    【学术界常用策略】Point Adjustment (PA)
    如果在一段连续的异常区间内检测到任意一个异常点，
    则认为该段异常被正确召回。
    """
    y_pred_pa = y_pred.copy()

    # 找到真实标签中所有的连续异常片段
    # 使用 diff 找到状态变化点 (0->1 或 1->0)
    diff = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        # 检查该片段内是否有预测为 1 的点
        if np.sum(y_pred[s:e]) > 0:
            y_pred_pa[s:e] = 1  # 修正整个片段为异常

    return y_pred_pa


def save_detailed_results(params, results, file_path='evaluation_logs.csv'):
    """保存运行参数和结果到 CSV"""
    file_exists = os.path.isfile(file_path)

    # 组合记录数据
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # 自动获取的参数
        'k_threshold': params.get('k', 'N/A'),
        'lambda_mse': params.get('lambda_mse', 'N/A'),
        'lambda_pred': params.get('lambda_pred', 'N/A'),
        'window_size': params.get('window_size', 'N/A'),
        'adj_threshold': params.get('adj_threshold', 'N/A'),
        # 原始结果指标 (Raw Metrics)
        'precision': round(results['precision'], 4),
        'recall': round(results['recall'], 4),
        'f1_score': round(results['f1'], 4),
        # PA 结果指标 (Point Adjustment) - 可选
        'f1_pa': round(results.get('f1_pa', 0), 4),
        # 混淆矩阵
        'TN': results['confusion_matrix'][0, 0],
        'FP': results['confusion_matrix'][0, 1],
        'FN': results['confusion_matrix'][1, 0],
        'TP': results['confusion_matrix'][1, 1],
    }

    headers = record.keys()

    # 确保文件未被占用
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
        print(f"\n[CSV Log] 已保存至: {file_path}")
    except PermissionError:
        print(f"\n[Error] 无法写入 {file_path}！请先关闭 Excel 文件。")


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
    """计算加权邻接矩阵"""
    print(f"\n[Graph] 正在构建加权邻接矩阵 (阈值={threshold})...")
    N = data.shape[-2]
    sample_data = data[:2000]
    reshaped = np.transpose(sample_data, (0, 1, 3, 2)).reshape(-1, N)
    corr_matrix = np.corrcoef(reshaped, rowvar=False)
    adj = np.where(np.abs(corr_matrix) >= threshold, np.abs(corr_matrix), 0.0)
    np.fill_diagonal(adj, 1.0)

    # 对称归一化
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return norm_adj.astype(np.float32)


def main():
    # 图构建阈值 (保持 0.7)
    ADJ_THRESHOLD = 0.7

    print(f"[Info] 评估模式启动. 设备: {tf.config.list_physical_devices('GPU')}")

    # 1. 数据加载
    data_path = 'data/processed/pamap2_caem.npy'
    if not os.path.exists(data_path):
        print("数据文件不存在，请先运行预处理。")
        return

    loaded = np.load(data_path, allow_pickle=True).item()
    X_segmented, y_segmented = segment_time_series(
        loaded['data'], loaded['label'], config.TIME_STEPS, 100
    )
    X_train, X_val, X_test, y_test = get_dataloaders(X_segmented, y_segmented, config)
    X_train, X_test = np.squeeze(X_train, axis=-1), np.squeeze(X_test, axis=-1)

    # 2. 模型构建与权重加载
    adj_matrix = get_adjacency_matrix(X_train, threshold=ADJ_THRESHOLD)
    model = build_caem_gcn_model(config, adj_matrix)

    checkpoint_path = 'checkpoints/gcn_caem_best_model.h5'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")

    model.load_weights(checkpoint_path)
    print("  SUCCESS: 权重加载成功!")

    # 3. 实例化 Trainer
    # 注意：这里实例化后，trainer.k 等属性就是你在 src/trainer.py 中设置的值
    trainer = CAEMTrainer(model, config)

    # 4. 自动提取参数 (核心修复：直接读对象属性)
    CURRENT_PARAMS = {
        'k': trainer.k,
        'lambda_mse': trainer.lambda_mse,
        'lambda_pred': trainer.lambda_pred,
        'window_size': trainer.window_size,
        'adj_threshold': ADJ_THRESHOLD
    }

    # 5. 执行评估 (Standard Metrics)
    trainer.calculate_threshold(X_train)
    results = trainer.evaluate(X_test, y_test)

    # 6. 计算 Point Adjustment Metrics (验证论文指标)
    y_pred_raw = (results['scores'] > trainer.threshold).astype(int)
    y_pred_pa = apply_point_adjustment(y_test, y_pred_raw)

    f1_pa = f1_score(y_test, y_pred_pa)
    precision_pa = precision_score(y_test, y_pred_pa)
    recall_pa = recall_score(y_test, y_pred_pa)

    # 将 PA 结果加入 results 字典以便保存
    results['f1_pa'] = f1_pa

    # 7. 打印与保存
    print("\n" + "=" * 50)
    print(f"     GCN-CAEM 评估结果 (自动同步参数)")
    print("=" * 50)
    print(f"当前生效参数: k={trainer.k}, window={trainer.window_size}, mse_weight={trainer.lambda_mse}")
    print("-" * 30)
    print("【原始指标 (Raw Metrics)】")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("-" * 30)
    print("【论文指标 (Point Adjustment)】<-- 关注这里")
    print(f"Precision (PA): {precision_pa:.4f}")
    print(f"Recall (PA):    {recall_pa:.4f}")
    print(f"F1 Score (PA):  {f1_pa:.4f}")
    print("=" * 50)

    save_detailed_results(CURRENT_PARAMS, results)


if __name__ == "__main__":
    main()