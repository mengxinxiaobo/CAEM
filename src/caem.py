# import tensorflow as tf
# from tensorflow.keras import layers, Model
# # 请确保这两个文件存在于 src 目录下
# from .networks import build_characterization_net
# from .layers import TemporalAttention, CAEMLossLayer
#
#
# def build_caem_model(config):
#     """
#     构建完整的 CAEM 模型
#     输入形状: (Batch, Memory_Window, N, T, 1)
#     """
#     N = config.NUM_SENSORS
#     T = config.TIME_STEPS
#     H = config.MEMORY_WINDOW
#
#     # 1. 模型输入: 序列化的传感器图像
#     input_seq = layers.Input(shape=(H, N, T, 1), name='seq_input')
#
#     # 2. 实例化 CAE
#     cae = build_characterization_net(N, T)
#
#     # 3. 逐时间步处理 (TimeDistributed Logic)
#     z_f_list = []
#     z_h_list = []  # z_h = [z_f, z_r]
#     x_recon_list = []
#     x_input_list = []
#
#     # 遍历 Window 中的每一个时间步
#     for t in range(H):
#         # 提取第 t 步的数据: (Batch, N, T, 1)
#         x_t = layers.Lambda(lambda x: x[:, t, :, :, :])(input_seq)
#         x_input_list.append(x_t)
#
#         # CAE 前向传播
#         z_f_t, x_recon_t = cae(x_t)
#
#         # 计算重构误差特征 z_r = (x - x')^2 (Eq. 3)
#         z_r_t = layers.Subtract()([x_t, x_recon_t])
#         z_r_t = layers.Lambda(lambda x: x ** 2)(z_r_t)
#
#         # 展平特征以便拼接
#         z_f_flat = layers.Flatten()(z_f_t)
#         z_r_flat = layers.Flatten()(z_r_t)
#
#         # 拼接 z_h = [z_f, z_r] (Eq. 6)
#         # 此时 z_h_t 的维度非常大 (约 13900 维)
#         z_h_t = layers.Concatenate()([z_f_flat, z_r_flat])
#
#         # 收集列表
#         z_f_list.append(z_f_t)
#         z_h_list.append(z_h_t)
#         x_recon_list.append(x_recon_t)
#
#     # 4. 准备 Memory Network 的输入和目标
#     # 将列表堆叠回 Tensor
#     # z_h_seq shape: (Batch, H, 13900)
#     z_h_seq = layers.Lambda(lambda x: tf.stack(x, axis=1))(z_h_list)
#
#     # Input to Memory Net: Past values [0 ... h-2]
#     # mem_input shape: (Batch, H-1, 13900)
#     mem_input = layers.Lambda(lambda x: x[:, :-1, :])(z_h_seq)
#
#     # Target for Prediction: Current value [h-1]
#     z_h_target = layers.Lambda(lambda x: x[:, -1, :])(z_h_seq)
#
#     # 获取当前时刻的其他数据用于 Loss 计算
#     z_f_current = z_f_list[-1]
#     x_recon_current = x_recon_list[-1]
#     x_current = x_input_list[-1]
#
#     # =========================================================
#     # 5. Memory Network Branch 1: Bi-LSTM + Attention (Non-linear)
#     # =========================================================
#     # 论文要求: sum bidirectional outputs
#     lstm_out = layers.Bidirectional(
#         layers.LSTM(config.HIDDEN_DIM, return_sequences=True),
#         merge_mode='sum'
#     )(mem_input)
#
#     context = TemporalAttention(config.HIDDEN_DIM)(lstm_out)
#
#     x = layers.Dropout(0.2)(context)
#     x = layers.Dense(1000, activation='relu')(x)
#     # 输出维度必须等于 z_h 的维度
#     y_h_pred = layers.Dense(z_h_target.shape[-1], name='y_h_pred')(x)
#
#     # =========================================================
#     # 6. Memory Network Branch 2: AR Model (Linear) -- [关键修复]
#     # =========================================================
#
#     # 1. 展平历史数据: (Batch, 55600)
#     ar_input = layers.Flatten()(mem_input)
#
#     # 2. 【核心修改】瓶颈层降维
#     # 先压缩到 128 维。这相当于提取了历史数据的核心线性特征。
#     # 参数量: 55600 * 128 ≈ 700万
#     ar_bottleneck = layers.Dense(128, activation='linear', name='ar_bottleneck')(ar_input)
#
#     # 3. 映射回目标维度
#     # 参数量: 128 * 13900 ≈ 170万
#     z_hat_pred = layers.Dense(z_h_target.shape[-1], name='z_hat_pred')(ar_bottleneck)
#
#
#
#     # =========================================================
#     # 7. 添加联合损失层
#     # =========================================================
#     # 这里 loss_out 实际上就是 x_current (因为我们改了 layers.py 做了透传)
#     # 把它命名为 x_current_with_loss 以示区分
#     x_current_with_loss = CAEMLossLayer(
#         lambda1=config.LAMBDA1,
#         lambda2=config.LAMBDA2,
#         lambda3=config.LAMBDA3
#     )([x_current, x_recon_current, z_f_current, z_h_target, y_h_pred, z_hat_pred])
#
#     # 8. 构建模型
#     # 注意：outputs 列表里的第 4 个元素，原来是 x_current，现在换成 x_current_with_loss
#     # 这样 CAEMLossLayer 就变成了模型输出的上游节点，绝对不会被剪枝了！
#     model = Model(
#         inputs=input_seq,
#         outputs=[y_h_pred, z_hat_pred, x_recon_current, x_current_with_loss, z_h_target]
#     )
#
#     return model





###################################################################################################

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# 导入配置与自定义组件
from src import config
from src.networks import build_gcn_characterization_net
from src.layers import TemporalAttention, CAEMLossLayer, GraphConv


def residual_gcn_block(x, units, adj, dropout_rate=0.2):
    """
    引入残差连接 (Residual Connection) 的图卷积块
    公式：H' = Activation(A * H * W) + H
    """
    # 路径 1: 标准图卷积路径
    h = GraphConv(units, activation='relu')([x, adj])
    h = layers.Dropout(dropout_rate)(h)

    # 路径 2: Shortcut 路径
    # 如果输入维度与输出维度不一致，使用 Dense 层进行线性变换以对齐维度
    shortcut = x
    if x.shape[-1] != units:
        shortcut = layers.Dense(units)(x)

    # 元素级相加实现残差连接
    return layers.Add()([h, shortcut])


def build_caem_gcn_model(config, adj_matrix):
    """
    构建基于升级版 GCN-BiLSTM 的 CAEM 模型

    主要改进点：
    1. 引入残差 GCN 块，防止深层特征退化。
    2. 时序预测分支由单向 LSTM 升级为双向 Bi-LSTM。
    """
    N = config.NUM_SENSORS
    T = config.TIME_STEPS
    H = config.MEMORY_WINDOW

    # 将邻接矩阵转为常量 Tensor，便于在模型内部多处共享
    adj_tensor = tf.constant(adj_matrix, dtype=tf.float32)

    # 模型输入形状: (Batch, H, N, T)
    input_seq = layers.Input(shape=(H, N, T), name='seq_input')

    # 实例化基础 GCN 编码器 (用于逐步提取空间特征)
    gcn_encoder = build_gcn_characterization_net(N, T, adj_tensor)

    z_f_list = []
    z_h_list = []
    x_recon_list = []
    x_input_list = []

    # 遍历滑动窗口中的每一个时间步
    for t in range(H):
        x_t = layers.Lambda(lambda x: x[:, t, :, :])(input_seq)
        x_input_list.append(x_t)

        # 获取基础图特征与重构结果
        z_f_t, x_recon_t = gcn_encoder(x_t)

        # 计算重构误差平方特征 z_r (用于增强异常判别力)
        z_r_t = layers.Subtract()([x_t, x_recon_t])
        z_r_t = layers.Lambda(lambda x: x ** 2)(z_r_t)

        # 展平特征以便拼接成综合隐变量 z_h
        z_f_flat = layers.Flatten()(z_f_t)
        z_r_flat = layers.Flatten()(z_r_t)

        # 拼接 z_h = [z_f, z_r]
        z_h_t = layers.Concatenate()([z_f_flat, z_r_flat])

        z_f_list.append(z_f_t)
        z_h_list.append(z_h_t)
        x_recon_list.append(x_recon_t)

    # 组织时序预测所需的序列数据
    z_h_seq = layers.Lambda(lambda x: tf.stack(x, axis=1))(z_h_list)
    mem_input = layers.Lambda(lambda x: x[:, :-1, :])(z_h_seq)  # 输入: 过去 h-1 步
    z_h_target = layers.Lambda(lambda x: x[:, -1, :])(z_h_seq)  # 目标: 当前第 h 步

    z_f_current = z_f_list[-1]
    x_recon_current = x_recon_list[-1]
    x_current = x_input_list[-1]

    # =========================================================
    # 5. 分支 1: Bi-LSTM + Attention (捕捉时空动态规律)
    # =========================================================
    # [升级] 改为双向 LSTM (Bidirectional)，并使用 concat 模式保留双向完整特征
    lstm_out = layers.Bidirectional(
        layers.LSTM(config.HIDDEN_DIM, return_sequences=True),
        merge_mode='concat'
    )(mem_input)

    # 注意：merge_mode='concat' 后，维度变为 HIDDEN_DIM * 2，Attention 层已适配
    context = TemporalAttention(config.HIDDEN_DIM * 2)(lstm_out)

    x = layers.Dropout(0.2)(context)
    x = layers.Dense(1024, activation='relu')(x)
    y_h_pred = layers.Dense(z_h_target.shape[-1], name='y_h_pred')(x)

    # =========================================================
    # 6. 分支 2: AR Model (捕捉线性趋势)
    # =========================================================
    ar_input = layers.Flatten()(mem_input)
    # 通过瓶颈层保留核心线性特征
    ar_bottleneck = layers.Dense(128, activation='linear', name='ar_bottleneck')(ar_input)
    z_hat_pred = layers.Dense(z_h_target.shape[-1], name='z_hat_pred')(ar_bottleneck)

    # =========================================================
    # 7. 添加联合损失层 (计算训练总 Loss)
    # =========================================================
    x_current_with_loss = CAEMLossLayer(
        lambda1=config.LAMBDA1,
        lambda2=config.LAMBDA2,
        lambda3=config.LAMBDA3
    )([x_current, x_recon_current, z_f_current, z_h_target, y_h_pred, z_hat_pred])

    # 8. 构建并返回最终模型
    model = Model(
        inputs=input_seq,
        outputs=[y_h_pred, z_hat_pred, x_recon_current, x_current_with_loss, z_h_target]
    )

    return model