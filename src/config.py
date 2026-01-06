# src/config.py

# --- 数据维度 ---
# 假设: 27个传感器, 100个时间步 (根据你的PyTorch示例)
NUM_SENSORS = 27       # N
TIME_STEPS = 100       # T
CHANNELS = 1           # Keras需要显式通道数

# --- 模型参数 ---
MEMORY_WINDOW = 5      # h (论文建议 h=5 或 10)
HIDDEN_DIM = 512       # Bi-LSTM 隐藏层维度
FEATURE_MAP_DIM = 64   # CAE 瓶颈层通道数

# --- 训练参数 ---
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

# --- 损失权重 (Eq. 20) ---
LAMBDA1 = 1e-4  # MMD Loss weight
LAMBDA2 = 0.5   # Linear Prediction weight
LAMBDA3 = 0.5   # Non-linear Prediction weight