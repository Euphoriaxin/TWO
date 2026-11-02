import torch
import torch.nn as nn
import torch.nn.functional as F


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, nhead=8, decoder_layers=3, dropout=0.1):
        super(TextDecoder, self).__init__()
        self.embed_dim = embed_dim

        # 词嵌入层+位置编码
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, embed_dim))

        # Transformer解码器（前置归一化）
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # 前置归一化
            ),
            num_layers=decoder_layers
        )

        # 正则化层
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, tgt):
        # tgt: [batch_size, seq_len]
        tgt_embed = self.embedding(tgt) * torch.sqrt(torch.tensor(self.embed_dim)).to(tgt.device)
        seq_len = tgt_embed.shape[1]
        tgt_embed = tgt_embed + self.pos_encoder[:, :seq_len, :]
        tgt_embed = self.dropout(tgt_embed)

        # 生成目标掩码（防止看到未来token）
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)

        # Transformer解码
        decoder_output = self.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(tgt == 0)
        )

        decoder_output = self.norm(decoder_output)
        return decoder_output