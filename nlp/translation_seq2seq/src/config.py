from pathlib import Path
# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据路径
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'

# 模型和日志路径
MODELS_DIR = ROOT_DIR / 'models'
LOG_DIR = ROOT_DIR / 'logs'

# 训练参数
SEQ_LEN = 32  # 输入序列长度
BATCH_SIZE = 128  # 批大小
EMBEDDING_DIM = 128  # 嵌入层维度
ENCODER_HIDDEN_SIZE = 256  # RNN 隐藏层维度
DECODER_HIDDEN_SIZE = 2 * ENCODER_HIDDEN_SIZE  # RNN 隐藏层维度
ENCODER_LAYERS = 2
DECODER_LAYERS = 1
LEARNING_RATE = 1e-3  # 学习率
EPOCHS = 30  # 训练轮数
