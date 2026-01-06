# # src/config.py
#
# # --- 数据维度 ---
# # 假设: 27个传感器, 100个时间步 (根据你的PyTorch示例)
# NUM_SENSORS = 27       # N
# TIME_STEPS = 100       # T
# CHANNELS = 1           # Keras需要显式通道数
#
# # --- 模型参数 ---
# MEMORY_WINDOW = 5      # h (论文建议 h=5 或 10)
# HIDDEN_DIM = 512       # Bi-LSTM 隐藏层维度
# FEATURE_MAP_DIM = 64   # CAE 瓶颈层通道数
#
# # --- 训练参数 ---
# BATCH_SIZE = 32
# EPOCHS = 100
# LEARNING_RATE = 1e-4
#
# # --- 损失权重 (Eq. 20) ---
# LAMBDA1 = 1e-4  # MMD Loss weight
# LAMBDA2 = 0.5   # Linear Prediction weight
# LAMBDA3 = 0.5   # Non-linear Prediction weight












import os

# ==============================================================================
# 1. 全局控制开关 (最常用修改项)
# ==============================================================================
# 当前使用的数据集名称
# 可选: 'PAMAP2', 'MSL' (火星车), 'SMAP' (卫星), 'SWaT' (水处理)
DATASET_NAME = 'MSL'

# ==============================================================================
# 2. 数据集维度自动适配
# ==============================================================================
if DATASET_NAME == 'PAMAP2':
    NUM_SENSORS = 27      # 原始维度
    TIME_STEPS = 100      # 动作识别通常窗口较短
elif DATASET_NAME == 'MSL':
    NUM_SENSORS = 55      # NASA MSL 官方维度
    TIME_STEPS = 100      # 工业/航天数据通常变化较慢，窗口可维持 100
elif DATASET_NAME == 'SMAP':
    NUM_SENSORS = 25      # NASA SMAP 官方维度
    TIME_STEPS = 100
elif DATASET_NAME == 'SWaT':
    NUM_SENSORS = 51      # 工业控制传感器
    TIME_STEPS = 50       # 采样率不同，有时需缩短窗口
else:
    raise ValueError(f"未知的 DATASET_NAME: {DATASET_NAME}")

# ==============================================================================
# 3. 模型架构参数 (GCN-CAEM 核心)
# ==============================================================================
# Bi-LSTM 隐藏层维度 (决定模型的容量)
# 复杂数据(如 MSL)建议 512 或 1024; 简单数据可降至 256
HIDDEN_DIM = 512

# 隐空间维度 (Z-Space, 瓶颈层)
LATENT_DIM = 128

# 卷积核大小 (仅当模型包含 CNN/Conv1D 分支时使用)
KERNEL_SIZE = 7

# Dropout 比率 (防止过拟合)
DROPOUT_RATE = 0.3

# GCN 是否使用残差连接 (建议 True)
USE_RESIDUAL = True

# ==============================================================================
# 4. 训练超参数
# ==============================================================================
BATCH_SIZE = 64       # 显存够大可开到 128，甚至 256
EPOCHS = 50           # 训练轮数 (通常 30-50 轮即可收敛)
LEARNING_RATE = 1e-4  # 学习率 (Adam 默认)

# 学习率衰减策略 (可选)
LR_DECAY_STEPS = 2000
LR_DECAY_RATE = 0.95

# ==============================================================================
# 5. 损失权重默认值 (仅作训练参考)
# 注意: 评估阶段的动态权重 (k, lambda) 已移交 Trainer 类动态管理
# ==============================================================================
# 训练时的初始权重配置
LAMBDA_MSE = 0.2      # 重构损失权重
LAMBDA_PRED = 1.8     # 预测损失权重
LAMBDA_KL = 0.01      # KL 散度权重 (如果使用 VAE 结构)

# ==============================================================================
# 6. 路径配置 (自动生成)
# ==============================================================================
# 自动根据数据集名称区分模型保存路径，防止覆盖
CHECKPOINT_DIR = f'checkpoints/{DATASET_NAME.lower()}'
LOG_DIR = f'logs/{DATASET_NAME.lower()}'
DATA_PATH = f'data/processed/nasa_{DATASET_NAME.lower()}_caem.npy'

# 确保目录存在
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)