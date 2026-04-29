import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import string
import random
from torch.utils.data import Dataset, DataLoader

class AdditionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_seq_len=16):
        super(AdditionTransformer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器 - 减少参数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # 从4倍减少到2倍
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 词嵌入 + 位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        
        # Transformer编码
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        
        # 输出投影
        output = self.output_projection(output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class AdditionDataset(Dataset):
    def __init__(self, num_samples=5000, max_digits=4):
        self.num_samples = num_samples
        self.max_digits = max_digits
        
        # 创建词汇表
        self.vocab = ['<pad>', '<sos>', '<eos>'] + list('0123456789') + ['+', '=']
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # 生成数据
        self.data = self._generate_data()
        
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # 生成随机数
            a = random.randint(1, 10**self.max_digits - 1)

            input_seq = str(a)
            target_seq = input_seq[::-1]

            data.append((input_seq, target_seq))
        return data
    
    def _encode_sequence(self, seq):
        return [self.char_to_idx.get(char, 0) for char in seq]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        
        # 编码序列
        input_ids = [self.char_to_idx['<sos>']] + self._encode_sequence(input_seq)
        target_ids = [self.char_to_idx['<sos>']] + self._encode_sequence(target_seq) + [self.char_to_idx['<eos>']]
        
        return torch.tensor(input_ids), torch.tensor(target_ids)

def collate_fn(batch):
    input_seqs, target_seqs = zip(*batch)
    
    # 填充序列
    input_seqs = nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=0)
    
    return input_seqs, target_seqs

def train_model():
    # 创建数据集
    dataset = AdditionDataset(num_samples=10000, max_digits=4)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)  # 增加batch_size
    
    # 创建模型
    model = AdditionTransformer(vocab_size=dataset.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    
    # 训练
    model.train()
    for epoch in range(50):  # 减少epoch数
        total_loss = 0
        for batch_idx, (input_seqs, target_seqs) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 创建输入和目标
            src = input_seqs
            tgt = target_seqs[:, 1:]   # 去掉第一个token作为目标
            
            # 前向传播
            output = model(src)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model, dataset

def predict(model, dataset, expression):
    """预测加法表达式的结果"""
    model.eval()
    
    # 编码输入
    input_seq = f"{expression}"
    input_ids = [dataset.char_to_idx['<sos>']] + dataset._encode_sequence(input_seq)
    input_tensor = torch.tensor(input_ids).unsqueeze(0)
    
    # 生成预测
    with torch.no_grad():
        # 使用贪心解码
        generated = input_tensor.clone()
        max_len = 20
        
        for _ in range(max_len):
            output = model(generated)
            next_token = output[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # 如果生成了结束符则停止
            if next_token.item() == dataset.char_to_idx['<eos>']:
                break
    
    # 解码结果
    result = ''.join([dataset.idx_to_char[idx.item()] for idx in generated[0][1:]])  # 去掉<sos>
    return result.replace('<eos>', '')

# 使用示例
if __name__ == "__main__":
    # 训练模型
    model, dataset = train_model()

    while True:
        # 测试预测
        test_data = str(input("请输入数字: "))
        prediction = predict(model, dataset, test_data)
        actual = test_data[::-1]  # 实际结果是输入的反转
        print(f"预测结果: {prediction}")
        print(f"实际结果: {actual}")
        print("-" * 30)



