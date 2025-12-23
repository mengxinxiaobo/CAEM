import tensorflow as tf
from tensorflow.keras import layers, Model


def build_characterization_net(num_sensors, time_steps):
    """
    对应 CharacterizationNetwork
    输入: (Batch, N, T, 1)
    """
    input_layer = layers.Input(shape=(num_sensors, time_steps, 1), name='cae_input')

    # --- Encoder ---
    # Layer 1
    x = layers.Conv2D(32, (4, 4), strides=1, padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Layer 2
    x = layers.Conv2D(64, (4, 4), strides=1, padding='same', activation='relu')(x)
    # 输出 z_f
    z_f = layers.MaxPooling2D((2, 2), padding='same', name='z_f_output')(x)

    # --- Decoder ---
    # Layer 3
    x = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(z_f)

    # Layer 4
    x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation='relu')(x)

    # Layer 5 (Output)
    x_recon = layers.Conv2DTranspose(1, (4, 4), strides=1, padding='same', name='x_recon_output')(x)

    # --- 维度强制对齐 (Interpolation trick equivalent) ---
    # 确保输出尺寸严格等于输入尺寸 (N, T)
    x_recon = layers.Resizing(num_sensors, time_steps)(x_recon)

    return Model(inputs=input_layer, outputs=[z_f, x_recon], name="CharacterizationNetwork")