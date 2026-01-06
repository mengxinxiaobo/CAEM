# import tensorflow as tf
# from tensorflow.keras import layers, Model
#
#
# def build_characterization_net(num_sensors, time_steps):
#     """
#     对应 CharacterizationNetwork
#     输入: (Batch, N, T, 1)
#     """
#     input_layer = layers.Input(shape=(num_sensors, time_steps, 1), name='cae_input')
#
#     # --- Encoder ---
#     # Layer 1
#     x = layers.Conv2D(32, (4, 4), strides=1, padding='same', activation='relu')(input_layer)
#     x = layers.MaxPooling2D((2, 2), padding='same')(x)
#
#     # Layer 2
#     x = layers.Conv2D(64, (4, 4), strides=1, padding='same', activation='relu')(x)
#     # 输出 z_f
#     z_f = layers.MaxPooling2D((2, 2), padding='same', name='z_f_output')(x)
#
#     # --- Decoder ---
#     # Layer 3
#     x = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(z_f)
#
#     # Layer 4
#     x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation='relu')(x)
#
#     # Layer 5 (Output)
#     x_recon = layers.Conv2DTranspose(1, (4, 4), strides=1, padding='same', name='x_recon_output')(x)
#
#     # --- 维度强制对齐 (Interpolation trick equivalent) ---
#     # 确保输出尺寸严格等于输入尺寸 (N, T)
#     x_recon = layers.Resizing(num_sensors, time_steps)(x_recon)
#
#     return Model(inputs=input_layer, outputs=[z_f, x_recon], name="CharacterizationNetwork")

#############################################################################################################
# 修改 src/networks.py

from tensorflow.keras import layers, Model, Input
from src.layers import GraphConv  # 导入刚才写的层


def build_gcn_characterization_net(num_sensors, time_steps, adj_matrix_tensor):
    """
    基于 GCN 的特征提取网络
    输入:
        - x: (Batch, N=27, T=100) -> 把时间 T 当作特征
        - adj: (N, N) -> 邻接矩阵
    输出:
        - z_f: (Batch, N, 64) -> 图特征
        - x_recon: (Batch, N, T) -> 重构数据
    """
    # 1. 输入定义
    # 注意: 这里输入不再是 (N, T, 1) 的图片，而是 (N, T) 的节点特征矩阵
    x_input = Input(shape=(num_sensors, time_steps), name='gcn_input')

    # 邻接矩阵作为常量输入 (Constant Input) 或者 第二个输入
    # 为了方便集成，我们这里假设 adj 是通过外部传入的 Tensor (常量)
    # 这样我们在 build model 时把它传进去即可

    # ----------------------------------------
    # Encoder (GCN)
    # ----------------------------------------
    # Layer 1: 捕捉浅层空间特征 (把 100 维时间特征 压缩到 64 维空间特征)
    h1 = GraphConv(64, activation='relu')([x_input, adj_matrix_tensor])
    h1 = layers.Dropout(0.2)(h1)

    # Layer 2: 捕捉深层空间特征
    z_f = GraphConv(32, activation='relu')([h1, adj_matrix_tensor])

    # z_f Shape: (Batch, 27, 32)
    # 这就是我们要送给 LSTM 的“图特征”

    # ----------------------------------------
    # Decoder (GCN 或 Dense) - 用于重构
    # ----------------------------------------
    # 为了对称，解码器也可以用 GCN，或者简单点用 Dense
    # 这里用 GraphConv 还原回 64
    h_decode = GraphConv(64, activation='relu')([z_f, adj_matrix_tensor])

    # 最后还原回 100 (原始时间步)
    # 注意最后一层通常不用 ReLU，因为原始数据可能是负的(归一化后)
    x_recon = GraphConv(time_steps, activation='linear')([h_decode, adj_matrix_tensor])

    return Model(inputs=x_input, outputs=[z_f, x_recon], name="GCN_Encoder")