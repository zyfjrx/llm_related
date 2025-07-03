from pathlib import Path
# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent

# 数据路径
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
PRETRAINED_MODELS_DIR = '/Users/zhangyf/llm/bert-base-chinese'

# 模型和日志路径
MODELS_DIR = ROOT_DIR / 'models'
LOG_DIR = ROOT_DIR / 'logs'

# 训练参数
SEQ_LEN = 64  # 输入序列长度
BATCH_SIZE = 32  # 批大小
LEARNING_RATE = 1e-3  # 学习率
EPOCHS = 30  # 训练轮数
NUM_CLASS = 30

