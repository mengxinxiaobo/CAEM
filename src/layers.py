import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


class TemporalAttention(layers.Layer):
    """
    对应论文 Eq. (14) - (17)
    """

    def __init__(self, hidden_dim, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.w_h = layers.Dense(hidden_dim)
        self.w_a = layers.Dense(1)

    def call(self, inputs):
        # inputs: Bi-LSTM output sequence (Batch, Seq_Len, Hidden_Dim)
        y_h = inputs

        # Eq. (14): M_h = tanh(W_h * y_h)
        m_h = tf.tanh(self.w_h(y_h))

        # Eq. (15): E_h = W_a * M_h
        e_h = self.w_a(m_h)  # (Batch, Seq, 1)

        # Eq. (16): A_h = softmax(E_h)
        a_h = tf.nn.softmax(e_h, axis=1)

        # Eq. (17): Context = sum(A_h * y_h)
        context = tf.reduce_sum(a_h * y_h, axis=1)  # (Batch, Hidden_Dim)

        return context


class CAEMLossLayer(layers.Layer):
    """
    计算联合损失 J(theta)
    对应论文 Eq. (20): MSE + lambda1*MMD + lambda2*LP + lambda3*NP
    """

    def __init__(self, lambda1, lambda2, lambda3, **kwargs):
        super(CAEMLossLayer, self).__init__(**kwargs)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def compute_mmd(self, source, target):
        """
        计算多核 MMD (Multi-kernel MMD)
        对应 PyTorch 版的 gaussian_kernel
        """
        # 展平特征: (Batch, Dim)
        source = tf.reshape(source, (tf.shape(source)[0], -1))
        target = tf.reshape(target, (tf.shape(target)[0], -1))

        # 拼接用于计算成对距离
        total = tf.concat([source, target], axis=0)
        total0 = tf.expand_dims(total, 0)  # (1, Total, Dim)
        total1 = tf.expand_dims(total, 1)  # (Total, 1, Dim)

        # L2 Distance Squared: ||x - y||^2
        l2_distance = tf.reduce_sum(tf.square(total0 - total1), axis=2)

        # Bandwidth heuristic
        n_samples = tf.cast(tf.shape(source)[0], tf.float32)
        bandwidth = tf.reduce_sum(l2_distance) / (4.0 * n_samples * n_samples)

        # 多尺度高斯核 (kernel_mul=2.0, kernel_num=5)
        kernel_val = 0.0
        # 避免 bandwidth 为 0
        bandwidth = tf.maximum(bandwidth, 1e-7)

        # 这里简化模拟 PyTorch 的循环: scale = [0.25, 0.5, 1, 2, 4]
        for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
            kernel_val += tf.exp(-l2_distance / (bandwidth * scale))

        # Extract blocks: XX + YY - 2XY
        n = tf.shape(source)[0]
        XX = kernel_val[:n, :n]
        YY = kernel_val[n:, n:]
        XY = kernel_val[:n, n:]

        return tf.reduce_mean(XX + YY - 2 * XY)

    def call(self, inputs):
        # 解包输入
        # x_true: 原始输入
        # x_recon: 重构输出
        # z_f: 卷积特征 (用于MMD)
        # z_h_target: 真实未来特征 (Label for Prediction)
        # y_h_pred: 非线性预测
        # z_hat_pred: 线性预测
        x_true, x_recon, z_f, z_h_target, y_h_pred, z_hat_pred = inputs

        # 1. Reconstruction Loss (MSE) - Eq. (3)
        loss_mse = tf.reduce_mean(tf.square(x_true - x_recon))

        # 2. MMD Loss - Eq. (4)-(5)
        # 采样目标分布 N(0, I)
        z_f_flat_shape = tf.shape(tf.reshape(z_f, (tf.shape(z_f)[0], -1)))
        target_dist = tf.random.normal(z_f_flat_shape)
        loss_mmd = self.compute_mmd(z_f, target_dist)

        # 3. Prediction Losses - Eq. (19)
        loss_np = tf.reduce_mean(tf.square(y_h_pred - z_h_target))
        loss_lp = tf.reduce_mean(tf.square(z_hat_pred - z_h_target))

        # 4. Total Loss - Eq. (20)
        total_loss = loss_mse + \
                     (self.lambda1 * loss_mmd) + \
                     (self.lambda2 * loss_lp) + \
                     (self.lambda3 * loss_np)

        self.add_loss(total_loss)

        # 添加监控指标 (Metrics)
        self.add_metric(loss_mse, name='mse')
        self.add_metric(loss_mmd, name='mmd')
        self.add_metric(loss_np, name='np_loss')
        self.add_metric(loss_lp, name='lp_loss')

        return x_true