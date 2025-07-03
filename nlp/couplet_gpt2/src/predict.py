import torch

import config
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def predict_batch(input_ids, model,tokenizer,  device):
    """
    :param input_ids: input_ids.shape[batch_size,seq_len] 中文
    :param model:
    :param en_tokenizer:
    :param zh_tokenizer:
    :param device:
    :return: 一批英文句子,e.g.: [[4,6,7],[11,23,45,78,99],[88,99,26,55]]
    """
    model.eval()
    with torch.no_grad():
        # 编码
        src_pad_mask = (input_ids == tokenizer.pad_token_id)
        # memory.shape[batch_size,src_len,d_model]
        memory = model.encode(input_ids,src_pad_mask)
        # 解码
        batch_size = input_ids.shape[0]
        # decoder_input.shape[batch_size,1]
        decoder_input = torch.full(size=(batch_size, 1), fill_value=tokenizer.sos_token_id, dtype=torch.long,
                                   device=device)
        generated = [[] for _ in range(batch_size)]
        is_finished = [False for _ in range(batch_size)]
        for t in range(1, config.SEQ_LEN):
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
            tgt_pad_mask = (decoder_input == tokenizer.pad_token_id)
            # output.shape[batch_size, tgt_len,en_vocab_size]
            output = model.decode(decoder_input, memory,tgt_mask,tgt_pad_mask,src_pad_mask)
            # last_output.shape[batch_size, en_vocab_size]
            last_output = output[:, -1, :]

            # last_output.shape[batch_size]
            predict_indexes = torch.argmax(last_output, dim=-1)
            # 处理每个时间步预测结果
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                else:
                    if predict_indexes[i].item() == tokenizer.eos_token_id:
                        is_finished[i] = True
                    else:
                        generated[i].append(predict_indexes[i].item())
            if all(is_finished):
                break
            decoder_input = torch.cat([decoder_input, predict_indexes.unsqueeze(1)], dim=1)

    return generated


def predict(user_input, model, tokenizer, device):
    # 处理数据
    input_ids = tokenizer.encode(user_input, config.SEQ_LEN, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    output_ids = predict_batch(input_ids, model, tokenizer, device)
    result = tokenizer.decode(output_ids[0])
    return result


def run_predict():
    device = torch.device("cpu")

    # tokenizer
    tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 模型
    model =  TranslationModel(vocab_size=tokenizer.vocab_size,padding_idx=tokenizer.pad_token_id)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    model.to(device)

    print("中英翻译：(输入q或者quit推出)")
    while True:
        user_input = input("上联>: ")
        if user_input in ['q', 'quit']:
            print("程序已退出")
            break
        if user_input.strip() == "":
            continue
        result = predict(user_input, model, tokenizer, device)
        print(f'下联>:{result}')


if __name__ == '__main__':
    run_predict()
