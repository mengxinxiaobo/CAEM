import tensorflow as tf
from tensorflow.keras import layers, Model
from .networks import build_characterization_net
from .layers import TemporalAttention, CAEMLossLayer


def build_caem_model(config):
    """
    构建完整的 CAEM 模型
    输入形状: (Batch, Memory_Window, N, T, 1)
    """
    N = config.NUM_SENSORS
    T = config.TIME_STEPS
    H = config.MEMORY_WINDOW

    # 1. 模型输入: 序列化的传感器图像
    input_seq = layers.Input(shape=(H, N, T, 1), name='seq_input')

    # 2. 实例化 CAE
    cae = build_characterization_net(N, T)

    # 3. 逐时间步处理 (TimeDistributed Logic)
    z_f_list = []
    z_h_list = []  # z_h = [z_f, z_r]
    x_recon_list = []
    x_input_list = []

    # 遍历 Window 中的每一个时间步
    for t in range(H):
        # 提取第 t 步的数据: (Batch, N, T, 1)
        x_t = layers.Lambda(lambda x: x[:, t, :, :, :])(input_seq)
        x_input_list.append(x_t)

        # CAE 前向传播
        z_f_t, x_recon_t = cae(x_t)

        # 计算重构误差特征 z_r = (x - x')^2 (Eq. 3)
        z_r_t = layers.Subtract()([x_t, x_recon_t])
        z_r_t = layers.Lambda(lambda x: x ** 2)(z_r_t)

        # 展平特征以便拼接
        z_f_flat = layers.Flatten()(z_f_t)
        z_r_flat = layers.Flatten()(z_r_t)

        # 拼接 z_h = [z_f, z_r] (Eq. 6)
        z_h_t = layers.Concatenate()([z_f_flat, z_r_flat])

        # 收集列表
        z_f_list.append(z_f_t)
        z_h_list.append(z_h_t)
        x_recon_list.append(x_recon_t)

    # 4. 准备 Memory Network 的输入和目标
    # 将列表堆叠回 Tensor
    z_h_seq = layers.Lambda(lambda x: tf.stack(x, axis=1))(z_h_list)  # (Batch, H, Dim)

    # Input to Memory Net: Past values [0 ... h-2]
    mem_input = layers.Lambda(lambda x: x[:, :-1, :])(z_h_seq)

    # Target for Prediction: Current value [h-1]
    z_h_target = layers.Lambda(lambda x: x[:, -1, :])(z_h_seq)

    # 获取当前时刻的其他数据用于 Loss 计算
    z_f_current = z_f_list[-1]
    x_recon_current = x_recon_list[-1]
    x_current = x_input_list[-1]

    # 5. Memory Network Branch 1: Bi-LSTM + Attention (Non-linear)
    # 论文要求: sum bidirectional outputs
    lstm_out = layers.Bidirectional(
        layers.LSTM(config.HIDDEN_DIM, return_sequences=True),
        merge_mode='sum'
    )(mem_input)

    context = TemporalAttention(config.HIDDEN_DIM)(lstm_out)

    x = layers.Dropout(0.2)(context)
    x = layers.Dense(1000, activation='relu')(x)
    # 输出维度必须等于 z_h 的维度
    y_h_pred = layers.Dense(z_h_target.shape[-1], name='y_h_pred')(x)

    # 6. Memory Network Branch 2: AR Model (Linear)
    # 输入: 展平的历史序列
    ar_input = layers.Flatten()(mem_input)
    z_hat_pred = layers.Dense(z_h_target.shape[-1], name='z_hat_pred')(ar_input)

    # 7. 添加联合损失层
    # 这里 loss_out 实际上就是 x_current (因为我们改了 layers.py 做了透传)
    # 把它命名为 x_current_with_loss 以示区分
    x_current_with_loss = CAEMLossLayer(
        lambda1=config.LAMBDA1,
        lambda2=config.LAMBDA2,
        lambda3=config.LAMBDA3
    )([x_current, x_recon_current, z_f_current, z_h_target, y_h_pred, z_hat_pred])

    # 8. 构建模型
    # 注意：outputs 列表里的第 4 个元素，原来是 x_current，现在换成 x_current_with_loss
    # 这样 CAEMLossLayer 就变成了模型输出的上游节点，绝对不会被剪枝了！
    model = Model(
        inputs=input_seq,
        outputs=[y_h_pred, z_hat_pred, x_recon_current, x_current_with_loss, z_h_target]
    )

    return model