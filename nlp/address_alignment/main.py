import config
from model import AddressTagging,load_params
from preprocess import AddressTaggingProcessor
from train import AddressTaggingTrainer
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
device = config.DEVICE
learning_rate = 1e-5
def model_go(train=None, test=None, inference=None, model_params_path=None):
    model = AddressTagging(model_name=config.PRETRAINED_DIR , label_list=config.LABELS)
    processor = AddressTaggingProcessor(
        data_path=config.RAW_DATA_DIR / 'data.txt',
        save_dir=config.PROCESSED_DATA_DIR,
        tokenizer=model.tokenizer,
        batch_size=config.BATCH_SIZE,
        label_list=config.LABELS,
    )
    trainer = AddressTaggingTrainer(model, device, 10, learning_rate)
    writer = None
    save_name = f"address_tagging-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    load_params(model, model_params_path)

    if train:
        writer = SummaryWriter(log_dir=config.LOGS_DIR  / save_name)
        dataloader = {
            "train": processor.get_dataloader("train"),
            "valid": processor.get_dataloader("valid"),
        }
        model_params_path = config.FINETUNED_DIR / f"{save_name}.pt"
        trainer(dataloader, model_params_path, writer)

    if test:
        test_dataloader = processor.get_dataloader("test")
        trainer({"test": test_dataloader},writer=writer,is_test=True)

    if writer:
        writer.close()



if __name__ == '__main__':
    model_go(train=True)