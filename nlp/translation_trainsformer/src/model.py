import torch
import torch.nn as nn
import config
import math


class PositionEncoding(nn.Module):
    def __init__(self, dim_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, dim_model, dtype=torch.float32)
        # pos.shape[max_len, 1]
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_trem = torch.exp(torch.arange(0, dim_model, 2, dtype=torch.float32) * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(pos * div_trem)
        pe[:, 1::2] = torch.cos(pos * div_trem)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x.shape[batch_size, seq_len, dim_model]
        #self.pe[:x.shape[1]].shape[seq_len, dim_model]
        return x + self.pe[:x.shape[1]]

# class PositionEncoding(nn.Module):
#     def __init__(self, dim_model, max_len=100):
#         super().__init__()
#         pe = torch.zeros(max_len, dim_model, dtype=torch.float32)
#         for pos in range(max_len):
#             for _2i in range(0, dim_model, 2):
#                 pe[pos, _2i] = math.sin(pos / (10000.0 ** (_2i / dim_model)))
#                 pe[pos, _2i + 1] = math.cos(pos / (10000.0 ** (_2i / dim_model)))
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         # x.shape[batch_size, seq_len, dim_model]
#         # self.pe[:x.shape[1]].shape[seq_len, dim_model]
#         return x + self.pe[:x.shape[1]]


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_idx, en_padding_idx):
        super().__init__()
        self.src_embedding = nn.Embedding(
            num_embeddings=zh_vocab_size,
            embedding_dim=config.DIM_MODEL,
            padding_idx=zh_padding_idx)
        self.tgt_embedding = nn.Embedding(
            num_embeddings=en_vocab_size,
            embedding_dim=config.DIM_MODEL,
            padding_idx=en_padding_idx)
        self.position_encoding = PositionEncoding(config.DIM_MODEL)

        self.transformer = nn.Transformer(
            d_model=config.DIM_MODEL,
            nhead=config.NUM_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            activation=config.ACTIVATION,
            batch_first=True,
        )
        self.linear = nn.Linear(config.DIM_MODEL, en_vocab_size)

    def encode(self, src, src_pad_mask):
        # src.shape[batch_size, seq_len]
        src_embed = self.src_embedding(src)
        # src_embed.shape[batch_size, seq_len,d_model]
        src_embed = self.position_encoding(src_embed)
        memory = self.transformer.encoder(src=src_embed, src_key_padding_mask=src_pad_mask)
        # memory.shape[batch_size, seq_len,d_model]
        return memory

    def decode(self, tgt, memory, tgt_mask, tgt_pad_mask, memory_pad_mask):
        tgt_embed = self.tgt_embedding(tgt)
        # tgt_embed.shape[batch_size, seq_len,d_model]
        tgt_embed = self.position_encoding(tgt_embed)
        # output.shape[batch_size, seq_len,d_model]
        output = self.transformer.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        # output.shape[batch_size, seq_len, en_vocab_size]
        output = self.linear(output)
        return output
    def forward(self, src, tgt,src_pad_mask, tgt_mask, tgt_pad_mask):
        memory = self.encode(src, src_pad_mask)
        output = self.decode(tgt, memory, tgt_mask, tgt_pad_mask, src_pad_mask)
        return output




if __name__ == '__main__':
    # pose = PositionEncoding(512, 128)
    # pose(torch.randn(32, 128, 512))
    model = TranslationModel(zh_vocab_size=10000, en_vocab_size=10000, zh_padding_idx=0, en_padding_idx=0)
    print(model)
