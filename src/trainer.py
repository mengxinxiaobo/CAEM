import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import tensorflow as tf


class CAEMTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.threshold = None
        self.mu = None
        self.sigma = None

    def calculate_anomaly_score(self, outputs):
        """
        计算每个样本的异常分数
        Score = MSE + lambda2*LP + lambda3*NP
        """
        y_h_pred, z_hat_pred, x_recon, x_current, z_h_target = outputs

        # 1. MSE (Reconstruction)
        # reduce_mean over (N, T, 1)
        loss_mse = np.mean((x_current - x_recon) ** 2, axis=(1, 2, 3))

        # 2. Prediction Errors
        loss_lp = np.mean((z_hat_pred - z_h_target) ** 2, axis=1)
        loss_np = np.mean((y_h_pred - z_h_target) ** 2, axis=1)

        # 3. Weighted Sum
        scores = loss_mse + \
                 (self.config.LAMBDA2 * loss_lp) + \
                 (self.config.LAMBDA3 * loss_np)

        return scores

    def calculate_threshold(self, train_data):
        """
        对应论文 Algorithm 1: Threshold Calculation
        THR = Mean + Std
        """
        print("Calculating Threshold on Training Data...")
        # 预测
        outputs = self.model.predict(train_data, batch_size=self.config.BATCH_SIZE)

        # 计算分数
        scores = self.calculate_anomaly_score(outputs)

        # 统计
        self.mu = np.mean(scores)
        self.sigma = np.std(scores)
        self.threshold = self.mu + self.sigma  # 严格遵循论文 (可改为 self.mu + k * self.sigma)

        print(f"Stats: mu={self.mu:.5f}, sigma={self.sigma:.5f}")
        print(f"Threshold set to: {self.threshold:.5f}")

    def evaluate(self, test_data, test_labels):
        """
        对应论文 Algorithm 1: Inference
        """
        if self.threshold is None:
            raise ValueError("Run calculate_threshold first!")

        print("Evaluating on Test Data...")
        outputs = self.model.predict(test_data, batch_size=self.config.BATCH_SIZE)

        scores = self.calculate_anomaly_score(outputs)

        # 判定
        preds = (scores > self.threshold).astype(int)

        # 指标计算
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, average='binary'
        )
        cm = confusion_matrix(test_labels, preds)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "raw_scores": scores
        }