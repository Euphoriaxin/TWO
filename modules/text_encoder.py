import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, nhead=8, encoder_layers=3, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.embed_dim = embed_dim

        # 词嵌入层+位置编码
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, embed_dim))

        # Transformer编码器（增加层归一化）
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # 前置归一化，提升稳定性
            ),
            num_layers=encoder_layers
        )

        # 输出层正则化
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, seq_len]
        src_embed = self.embedding(src) * torch.sqrt(torch.tensor(self.embed_dim)).to(src.device)
        seq_len = src_embed.shape[1]
        src_embed = src_embed + self.pos_encoder[:, :seq_len, :]
        src_embed = self.dropout(src_embed)

        # Transformer编码
        encoder_output = self.encoder(src_embed)
        encoder_output = self.norm(encoder_output)

        return encoder_output