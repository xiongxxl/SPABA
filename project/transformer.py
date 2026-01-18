import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict


# 假设我们使用字符级tokenizer
def build_vocab():
    charset = list("CNOPSHFBrI[]()=#123456789%@+-\\/.") + ["<pad>", "<unk>"]
    vocab = {ch: i for i, ch in enumerate(charset)}
    return vocab


# 定义SMILES字符串的tokenizer
def smiles_tokenizer(smiles, vocab, max_length=256):
    # 将SMILES转为token ID
    token_ids = [vocab.get(ch, vocab["<unk>"]) for ch in smiles[:max_length]]
    pad_length = max_length - len(token_ids)
    return token_ids + [vocab["<pad>"]] * pad_length


# 示例输入数据
vocab = build_vocab()
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
max_length = 256

# 将SMILES字符串转换为token ID
input_ids = smiles_tokenizer(smiles, vocab, max_length)
print(input_ids)


class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, num_labels, max_length=256, dropout=0.1):
        super(SMILESTransformer, self).__init__()
        # 输入嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, embed_dim))

        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 多标签分类器（输出n维标签）
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, num_labels)
        )

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer_encoder(x)  # (batch_size, seq_len, embed_dim)
        x = x.mean(dim=1)  # 使用序列池化（mean pooling）
        logits = self.classifier(x)  # (batch_size, num_labels)
        return logits


# 示例：多标签目标
# 目标是一个多标签向量，例如：
# [0, 1, 0, 1] 表示有多个标签，二进制形式表示是否属于该标签。
labels = torch.tensor([
    [0, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=torch.float)

# 设置超参数
vocab_size = len(vocab)
embed_dim = 128
num_heads = 4
num_layers = 2
ff_dim = 256
num_labels = 4  # 例如4个标签进行分类
max_length = 256

# 初始化模型
model = SMILESTransformer(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_dim=ff_dim,
    num_labels=num_labels,
    max_length=max_length
)

# 使用BCEWithLogitsLoss损失函数，适合多标签分类任务
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 模拟训练数据
input_ids = torch.tensor([input_ids, input_ids])  # 输入的batch数据

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    logits = model(input_ids)

    # 计算损失
    loss = criterion(logits, labels)

    # 反向传播
    loss.backward()

    # 优化器更新参数
    optimizer.step()

    # 打印损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_smiles = "CCC(=O)O"
    test_input_ids = smiles_tokenizer(test_smiles, vocab, max_length)
    test_input_ids = torch.tensor([test_input_ids])  # 扩展为batch的形式

    output_logits = model(test_input_ids)
    predicted_labels = torch.sigmoid(output_logits)  # 使用sigmoid函数得到每个标签的概率

    # 将概率转换为0或1标签
    predicted_labels = (predicted_labels > 0.5).float()
    print(predicted_labels)

