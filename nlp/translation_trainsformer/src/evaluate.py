import torch
from tqdm import tqdm
import config
from dataset import get_dataloader
from model import TranslationEncoder, TranslationDecoder
from predict import predict_batch
from tokenizer import ChineseTokenizer, EnglishTokenizer
from nltk.translate.bleu_score import corpus_bleu

def evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device):
    references = [] # [[[4,6,7][,[[11,23,45,78,99]],[[88,99,26,55]] .....]
    predictions = [] # [[4,6,7],[11,23,45,78,99],[88,99,26,55] .....]
    special_tokens = [en_tokenizer.pad_token_id, en_tokenizer.eos_token_id, en_tokenizer.sos_token_id]
    for inputs, targets in tqdm(dataloader, desc='evaluate'):
        # inputs.shape[batch_size,seq_len]
        inputs = inputs.to(device)
        # 真实译文 targets.shape[batch_size,seq_len]
        targets = targets.tolist()
        # 预测译文 [[4,6,7],[11,23,45,78,99],[88,99,26,55] .....]
        batch_result = predict_batch(inputs, encoder, decoder, zh_tokenizer, en_tokenizer, device)
        predictions.extend(batch_result)
        references.extend([[[index for index in target if index not in special_tokens]] for target in targets])
    return corpus_bleu(references, predictions)




def run_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab_zh.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab_en.txt')

    # 模型
    encoder = TranslationEncoder(vocab_size=zh_tokenizer.vocab_size,
                                 padding_idx=zh_tokenizer.pad_token_id)
    encoder.load_state_dict(torch.load(config.MODELS_DIR / 'encoder_model.pt'))
    encoder.to(device)
    decoder = TranslationDecoder(vocab_size=en_tokenizer.vocab_size,
                                 padding_idx=en_tokenizer.pad_token_id)
    decoder.load_state_dict(torch.load(config.MODELS_DIR / 'decoder_model.pt'))
    decoder.to(device)

    # 加载数据集
    dataloader = get_dataloader(train=False)
    bleu = evaluate(dataloader,encoder,decoder,zh_tokenizer,en_tokenizer, device)
    print(f"BLEU: {bleu:.4f}")


if __name__ == '__main__':
    run_evaluate()