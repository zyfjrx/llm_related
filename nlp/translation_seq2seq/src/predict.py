import torch

import config
from model import TranslationEncoder, TranslationDecoder
from tokenizer import ChineseTokenizer,EnglishTokenizer


def predict_batch(input_ids, encoder, decoder, en_tokenizer, zh_tokenizer, device):
    """

    :param input_ids: input_ids.shape[batch_size,seq_len] 中文
    :param encoder:
    :param decoder:
    :param en_tokenizer:
    :param zh_tokenizer:
    :param device:
    :return: 一批英文句子,e.g.: [[4,6,7],[11,23,45,78,99],[88,99,26,55]]
    """
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # context_vector.shape[batch_size,decoder_hidden_size]
        context_vector = encoder(input_ids)
        batch_size = input_ids.shape[0]
        # decoder_input.shape[batch_size,1]
        decoder_input = torch.full(size=(batch_size, 1), fill_value=zh_tokenizer.sos_token_id, dtype=torch.long,device=device)
        # decoder_hidden.shape[1, batch_size, decoder_hidden_size]
        decoder_hidden = context_vector.unsqueeze(0)

        generated = [[] for _ in range(batch_size)]
        is_finished = [False for _ in range(batch_size)]
        for t in range(1, config.SEQ_LEN):
            # decoder_output.shape[batch_size, 1, vocab_size]
            decoder_output,decoder_hidden = decoder(decoder_input, decoder_hidden)
            # predict_indexes.shape[batch_size, 1]
            predict_indexes = torch.argmax(decoder_output, dim=-1)
            # 处理每个时间步预测结果
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                else:
                    if predict_indexes[i].item() == en_tokenizer.eos_token_id:
                        is_finished[i] = True
                    else:
                        generated[i].append(predict_indexes[i].item())
            if all(is_finished):
                break
            decoder_input = predict_indexes

    return  generated






def predict(user_input, encoder, decoder, en_tokenizer, zh_tokenizer, device):
    # 处理数据
    input_ids = zh_tokenizer.encode(user_input,config.SEQ_LEN,add_special_tokens=False)
    input_ids = torch.tensor([input_ids],dtype=torch.long).to(device)
    output_ids = predict_batch(input_ids,encoder, decoder, en_tokenizer, zh_tokenizer, device)
    result = en_tokenizer.decode(output_ids[0])
    return result


def run_predict():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

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

    print("中英翻译：(输入q或者quit推出)")
    while True:
        user_input = input("中文> ")
        if user_input in ['q', 'quit']:
            print("程序已退出")
            break
        if user_input.strip() == "":
            continue
        result = predict(user_input,encoder,decoder,en_tokenizer,zh_tokenizer,device)
        print(f'英文：{result}')

if __name__ == '__main__':
    run_predict()