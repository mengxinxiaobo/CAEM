# import numpy as np
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
# import tensorflow as tf
#
#
# class CAEMTrainer:
#     def __init__(self, model, config):
#         self.model = model
#         self.config = config
#         self.threshold = None
#         self.mu = None
#         self.sigma = None
#
#     def calculate_anomaly_score(self, outputs):
#         """
#         计算每个样本的异常分数
#         Score = MSE + lambda2*LP + lambda3*NP
#         """
#         y_h_pred, z_hat_pred, x_recon, x_current, z_h_target = outputs
#
#         # 1. MSE (Reconstruction)
#         # reduce_mean over (N, T, 1)
#         loss_mse = np.mean((x_current - x_recon) ** 2, axis=(1, 2, 3))
#
#         # 2. Prediction Errors
#         loss_lp = np.mean((z_hat_pred - z_h_target) ** 2, axis=1)
#         loss_np = np.mean((y_h_pred - z_h_target) ** 2, axis=1)
#
#         # 3. Weighted Sum
#         scores = loss_mse + \
#                  (self.config.LAMBDA2 * loss_lp) + \
#                  (self.config.LAMBDA3 * loss_np)
#
#         return scores
#
#     def calculate_threshold(self, train_data):
#         """
#         对应论文 Algorithm 1: Threshold Calculation
#         THR = Mean + Std
#         """
#         print("Calculating Threshold on Training Data...")
#         # 预测
#         outputs = self.model.predict(train_data, batch_size=self.config.BATCH_SIZE)
#
#         # 计算分数
#         scores = self.calculate_anomaly_score(outputs)
#
#         # 统计
#         self.mu = np.mean(scores)
#         self.sigma = np.std(scores)
#         self.threshold = self.mu + self.sigma  # 严格遵循论文 (可改为 self.mu + k * self.sigma)
#
#         print(f"Stats: mu={self.mu:.5f}, sigma={self.sigma:.5f}")
#         print(f"Threshold set to: {self.threshold:.5f}")
#
#     def evaluate(self, test_data, test_labels):
#         """
#         对应论文 Algorithm 1: Inference
#         """
#         if self.threshold is None:
#             raise ValueError("Run calculate_threshold first!")
#
#         print("Evaluating on Test Data...")
#         outputs = self.model.predict(test_data, batch_size=self.config.BATCH_SIZE)
#
#         scores = self.calculate_anomaly_score(outputs)
#
#         # 判定
#         preds = (scores > self.threshold).astype(int)
#
#         # 指标计算
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             test_labels, preds, average='binary'
#         )
#         cm = confusion_matrix(test_labels, preds)
#
#         return {
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "confusion_matrix": cm,
#             "raw_scores": scores
#         }


import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf


class CAEMTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # =========================================================
        # 【核心修改】将超参数集中管理 (改这里，评估代码会自动同步)
        # =========================================================
        self.k = 2  # 阈值系数 (建议 0.7 ~ 1.0)
        self.window_size = 7  # 平滑窗口大小 (建议 5, 7, 9)
        self.lambda_mse = 0.1  # 重构误差权重 (降权以减少误报)
        self.lambda_pred = 1.9  # 预测误差权重 (加权以利用 Bi-LSTM)
        # =========================================================

        self.threshold = None
        self.threshold_mean = None
        self.threshold_std = None

    def calculate_anomaly_score(self, model_outputs):
        """
        计算综合异常分数
        outputs: [y_h_pred, z_hat_pred, x_recon, loss_out, z_h_target]
        """
        y_h_pred = model_outputs[0]
        z_hat_pred = model_outputs[1]
        x_recon = model_outputs[2]
        x_current = model_outputs[3]  # Loss层透传的真实输入
        z_h_target = model_outputs[4]

        # 1. 重构误差 (MSE)
        diff = x_current - x_recon
        if len(diff.shape) == 4:  # CNN (Batch, N, T, 1)
            loss_mse = np.mean(diff ** 2, axis=(1, 2, 3))
        else:  # GCN (Batch, N, T)
            loss_mse = np.mean(diff ** 2, axis=(1, 2))

        # 2. 预测误差
        # 线性 (AR)
        diff_lp = z_h_target - z_hat_pred
        loss_lp = np.mean(diff_lp ** 2, axis=1)
        # 非线性 (Bi-LSTM)
        diff_np = z_h_target - y_h_pred
        loss_np = np.mean(diff_np ** 2, axis=1)

        # 3. 计算综合分数 (使用类属性 self.lambda_xxx)
        total_score = self.lambda_mse * loss_mse + self.lambda_pred * (loss_lp + loss_np)

        # 4. 滑动平滑 (使用类属性 self.window_size)
        if len(total_score) > self.window_size:
            smoothed_score = np.convolve(
                total_score,
                np.ones(self.window_size) / self.window_size,
                mode='same'
            )
            return smoothed_score

        return total_score

    def calculate_threshold(self, X_train):
        """
        使用训练集计算阈值
        Threshold = Mean + k * Std
        """
        print("Calculating Threshold on Training Data...")

        # 预测
        outputs = self.model.predict(X_train, batch_size=self.config.BATCH_SIZE, verbose=1)

        # 计算分数
        scores = self.calculate_anomaly_score(outputs)

        self.threshold_mean = np.mean(scores)
        self.threshold_std = np.std(scores)

        # 使用类属性 self.k
        self.threshold = self.threshold_mean + self.k * self.threshold_std

        print(f"[Info] Threshold calculated: {self.threshold:.4f} "
              f"(Mean: {self.threshold_mean:.4f}, Std: {self.threshold_std:.4f}, k={self.k})")

    def evaluate(self, X_test, y_test):
        if self.threshold is None:
            raise ValueError("Threshold not calculated! Please run calculate_threshold() first.")

        print("Evaluating on Test Data...")
        outputs = self.model.predict(X_test, batch_size=self.config.BATCH_SIZE, verbose=1)
        scores = self.calculate_anomaly_score(outputs)

        # 判定
        y_pred = (scores > self.threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'scores': scores
        }