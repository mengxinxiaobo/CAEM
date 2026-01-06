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
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class CAEMTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
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
        # model_outputs[3] 是联合 loss 层的输出 (即 x_current)
        z_h_target = model_outputs[4]

        # 获取真实输入值 (通过 Loss 层的透传)
        x_current = model_outputs[3]

        # =========================================================
        # 1. 重构误差 (MSE)
        # =========================================================
        diff = x_current - x_recon

        # 自动判断维度以适配 GCN (3维) 和 CNN (4维)
        if len(diff.shape) == 4:  # CNN 版本 (Batch, N, T, 1)
            loss_mse = np.mean(diff ** 2, axis=(1, 2, 3))
        else:  # GCN 版本 (Batch, N, T)
            loss_mse = np.mean(diff ** 2, axis=(1, 2))

        # =========================================================
        # 2. 预测误差 (Linear & Non-linear)
        # =========================================================
        # 线性预测误差 (AR Branch)
        diff_lp = z_h_target - z_hat_pred
        loss_lp = np.mean(diff_lp ** 2, axis=1)  # (Batch, )

        # 非线性预测误差 (LSTM Branch)
        diff_np = z_h_target - y_h_pred
        loss_np = np.mean(diff_np ** 2, axis=1)  # (Batch, )

        # =========================================================
        # 3. 计算综合分数 (关键修改区域)
        # =========================================================
        # 原策略: MSE 占主导
        # total_score = loss_mse + 0.5 * (loss_lp + loss_np)

        # [新策略]: 侧重于预测误差
        # 因为 GCN 的 MSE 普遍偏高 (约0.4)，如果不降权，它会淹没 LSTM 发现的预测异常。
        # 我们更相信 LSTM 对动作连贯性的判断。
        lambda_mse = 0.5  # 降低重构权重
        lambda_pred = 1.5  # 提高预测权重 (LP + NP)

        total_score = lambda_mse * loss_mse + lambda_pred * (loss_lp + loss_np)

        return total_score

    def calculate_threshold(self, X_train):
        """
        使用训练集(正常数据)计算阈值
        Threshold = Mean + k * Std
        """
        print("Calculating Threshold on Training Data...")

        outputs = self.model.predict(X_train, batch_size=self.config.BATCH_SIZE, verbose=1)

        scores = self.calculate_anomaly_score(outputs)

        self.threshold_mean = np.mean(scores)
        self.threshold_std = np.std(scores)

        # =========================================================
        # 阈值系数 k
        # =========================================================
        # 保持 k=0.2，配合新的权重策略使用
        k = 0.2

        self.threshold = self.threshold_mean + k * self.threshold_std

        print(
            f"[Info] Threshold calculated: {self.threshold:.4f} (Mean: {self.threshold_mean:.4f}, Std: {self.threshold_std:.4f}, k={k})")

    def evaluate(self, X_test, y_test):
        """
        在测试集上评估模型性能
        """
        if self.threshold is None:
            raise ValueError("Threshold not calculated! Please run calculate_threshold() first.")

        print("Evaluating on Test Data...")

        outputs = self.model.predict(X_test, batch_size=self.config.BATCH_SIZE, verbose=1)

        scores = self.calculate_anomaly_score(outputs)

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