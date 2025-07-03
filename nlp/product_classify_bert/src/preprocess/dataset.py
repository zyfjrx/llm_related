from enum import Enum
from datasets import load_from_disk
from conf import config
from torch.utils.data import DataLoader

class DataType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALID = "valid"


def get_dataset(type=DataType.TRAIN):
    dataset = load_from_disk(str(config.PROCESSED_DATA_DIR / type.value))
    return dataset
def get_dataloader(type=DataType.TRAIN):
    dataset = get_dataset(type)
    dataset.set_format('torch')
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

