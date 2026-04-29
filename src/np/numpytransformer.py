import numpy as np
import numpyNN as nn

np.set_printoptions(threshold=np.inf, linewidth=99999)

class ScaledDotProduct:
    def __init__(self, d_k):
        self.d_k = d_k
        self.softmax = nn.Softmax()
        self.scale = 1.0 / np.sqrt(d_k)  # 缩放因子

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask=None):
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

        # 计算注意力分数
        scores = np.matmul(Q, K.transpose(1, 0)) * self.scale

        if mask is not None:
            # 使用 np.where 和广播来应用掩码，将掩码为0的位置设置为一个很小的数
            scores = np.where(mask == 0, -1e9, scores)

        self.attention_weights = self.softmax.forward(scores)
    
        self.output = np.matmul(self.attention_weights, V)
        return self.output

    def backward(self, g_prev: np.ndarray):
        g_V = np.matmul(self.attention_weights.transpose(1, 0), g_prev)

        g_attention_weights = np.matmul(g_prev, self.V.transpose(1, 0))

        g_scores = self.softmax.backward(g_attention_weights)

        # 在反向传播到 Q 和 K 之前，应用掩码将对应位置的梯度清零
        if self.mask is not None:
            g_scores = np.where(self.mask == 0, 0, g_scores)

        g_Q = np.matmul(g_scores, self.K) * self.scale
        g_K = np.matmul(g_scores.transpose(1, 0), self.Q) * self.scale

        return g_Q, g_K, g_V

class SelfAttention:
    def __init__(self, d_model, d_k, d_v, optimizer):
        """
            d_model: 输入和输出的维度
            d_k: Key和Query的维度
            d_v: Value的维度
            lr: 学习率
        """
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.optimizer = optimizer
        
        # 线性变换层，用于生成Q、K、V
        self.W_q = nn.Linear(d_model, d_k, optimizer)
        self.W_k = nn.Linear(d_model, d_k, optimizer)
        self.W_v = nn.Linear(d_model, d_v, optimizer)
        self.W_o = nn.Linear(d_v, d_model, optimizer)
        
        # 缩放点积注意力
        self.attention = ScaledDotProduct(d_k)
    
    def forward(self, X, mask=None):
        """
            X: 输入张量，形状为 (seq_len, d_model)
            mask: 注意力掩码，可选
            output: 输出张量，形状为 (seq_len, d_model)
        """

        self.X = X

        self.Q = self.W_q.forward(X)
        self.K = self.W_k.forward(X) 
        self.V = self.W_v.forward(X)
        
        self.attention_output = self.attention.forward(self.Q, self.K, self.V, mask)

        self.output = self.W_o.forward(self.attention_output)
        
        return self.output
    
    def backward(self, g_prev):
        """
            g_prev: 来自上一层的梯度
        """
        # 输出线性层的反向传播
        g_attention_output = self.W_o.backward(g_prev)
        
        # 注意力机制的反向传播
        g_Q, g_K, g_V = self.attention.backward(g_attention_output)
        
        # Q、K、V线性层的反向传播
        g_X_q = self.W_q.backward(g_Q)
        g_X_k = self.W_k.backward(g_K)
        g_X_v = self.W_v.backward(g_V)

        return g_X_q, g_X_k, g_X_v

class MultiHeadAttention:
    def __init__(self, d_model, d_k, d_v, n_heads, optimizer):
        """
            d_model: 输入和输出的维度
            d_k: Key和Query的维度  
            d_v: Value的维度
            n_heads: 注意力头的数量
            lr: 学习率
        """
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.optimizer = optimizer
        
        # 每个头的线性变换层
        self.W_qs = nn.Linear(d_model, n_heads * d_k, optimizer)
        self.W_ks = nn.Linear(d_model, n_heads * d_k, optimizer)
        self.W_vs = nn.Linear(d_model, n_heads * d_v, optimizer)
        
        # 输出线性层
        self.W_o = nn.Linear(n_heads * d_v, d_model, optimizer)
        
        # 为每个头创建独立的注意力实例
        self.attentions = [ScaledDotProduct(d_k) for _ in range(n_heads)]
    
    def forward(self, Q, K, V, mask=None):
        self.Q = Q
        self.K = K
        self.V = V
        q_seq_len = Q.shape[0]
        k_seq_len, _ = K.shape
        v_seq_len, _ = V.shape

        # 生成Q、K、V并重塑为多头形式
        Qs = self.W_qs.forward(Q).reshape(q_seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        Ks = self.W_ks.forward(K).reshape(k_seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        Vs = self.W_vs.forward(V).reshape(v_seq_len, self.n_heads, self.d_v).transpose(1, 0, 2)
        
        # 为每个头应用独立的注意力
        attention_outputs = [
            self.attentions[i].forward(Qs[i], Ks[i], Vs[i], mask)
            for i in range(self.n_heads)
        ]
        
        # 拼接所有头的输出
        stacked_attention = np.stack(attention_outputs, axis=0)

        attention_output = stacked_attention.transpose(1, 0, 2).reshape(q_seq_len, self.n_heads * self.d_v)

        self.output = self.W_o.forward(attention_output)
        
        # 保存中间结果
        self.Qs = Qs
        self.Ks = Ks
        self.Vs = Vs
        self.attention_outputs = attention_outputs

        assert self.output.shape == (q_seq_len, self.d_model), "Output shape mismatch"
        
        return self.output
    
    def backward(self, g_prev):
        g_attention_output = self.W_o.backward(g_prev)
        
        q_seq_len = g_attention_output.shape[0]
        k_seq_len = self.K.shape[0]
        v_seq_len = self.V.shape[0]
        g_attention_heads = g_attention_output.reshape(q_seq_len, self.n_heads, self.d_v).transpose(1, 0, 2)
        
        # 为每个头计算梯度
        g_Qs, g_Ks, g_Vs = [], [], []
        for i in range(self.n_heads):
            g_Q, g_K, g_V = self.attentions[i].backward(g_attention_heads[i])
            g_Qs.append(g_Q)
            g_Ks.append(g_K)
            g_Vs.append(g_V)
        
        g_Qs_stacked = np.stack(g_Qs, axis=0)
        g_Ks_stacked = np.stack(g_Ks, axis=0)
        g_Vs_stacked = np.stack(g_Vs, axis=0)
        
        # 转置回原始形状并重塑，操作与前向传播的拆分相反
        g_Q_combined = g_Qs_stacked.transpose(1, 0, 2).reshape(q_seq_len, self.n_heads * self.d_k)
        g_K_combined = g_Ks_stacked.transpose(1, 0, 2).reshape(k_seq_len, self.n_heads * self.d_k)
        g_V_combined = g_Vs_stacked.transpose(1, 0, 2).reshape(v_seq_len, self.n_heads * self.d_v)
        
        g_X_q = self.W_qs.backward(g_Q_combined)
        g_X_k = self.W_ks.backward(g_K_combined)
        g_X_v = self.W_vs.backward(g_V_combined)
        
        return g_X_q, g_X_k, g_X_v


class Encoderlayer:
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff, optimizer):
        """
            d_model: 模型维度
            d_k: Key和Query的维度
            d_v: Value的维度  
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            lr: 学习率
        """
        self.d_model = d_model
        self.optimizer = optimizer
        
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads, self.optimizer)
        self.layer_norm_1 = nn.LayerNorm(d_model, optimizer)

        self.feed_forward = nn.FeedForward(d_model, d_ff, self.optimizer)
        self.layer_norm_2 = nn.LayerNorm(d_model, optimizer)

    def forward(self, X, padding_mask=None):
        self.X = X
        
        # 多头自注意力
        attention_output = self.self_attention.forward(X, X, X, padding_mask)
        
        # 残差连接和层归一化
        self.attention_output = self.layer_norm_1.forward(X + attention_output)
        
        # 前馈网络
        ff_output = self.feed_forward.forward(self.attention_output)
        
        # 残差连接和层归一化
        self.output = self.layer_norm_2.forward(self.attention_output + ff_output)
        
        return self.output
    
    def backward(self, g_prev):
        """
            g_prev: 来自上一层的梯度
            output: 对输入X的梯度
        """
        g_residual_2 = self.layer_norm_2.backward(g_prev)
        g_ff = self.feed_forward.backward(g_residual_2)
        g_attention_output = g_ff + g_residual_2

        # 第一个层归一化的反向传播
        g_residual_1 = self.layer_norm_1.backward(g_attention_output)
        # 第一个残差连接的反向传播
        a, b, c = self.self_attention.backward(g_residual_1)
        g_attention = a + b + c
        g_X = g_attention + g_residual_1
        return g_X
    
class DecoderLayer:
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff, optimizer):
        """
            d_model: 模型维度
            d_k: Key和Query的维度
            d_v: Value的维度  
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            lr: 学习率
        """
        self.d_model = d_model
        self.optimizer = optimizer
        
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads, optimizer)
        self.layer_norm_1 = nn.LayerNorm(d_model, optimizer)

        self.cross_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads, optimizer)
        self.layer_norm_2 = nn.LayerNorm(d_model, optimizer)

        self.feed_forward = nn.FeedForward(d_model, d_ff, optimizer)
        self.layer_norm_3 = nn.LayerNorm(d_model, optimizer)

    def forward(self, X, encoder_output, lookahead_mask, decoder_padding_mask, encoder_padding_mask):
        """
            X: 解码器输入，形状为 (seq_len, d_model)
            encoder_output: 编码器输出，形状为 (seq_len, d_model)
            mask: 注意力掩码，可选
            output: 解码器输出，形状为 (seq_len, d_model)
        """
        self.X = X
        self.encoder_output = encoder_output
        
        # 自注意力: 合并 lookahead_mask 和 decoder_padding_mask
        self_attention_mask = lookahead_mask
        if decoder_padding_mask is not None:
            # decoder_padding_mask 的形状是 (seq_len,)
            # lookahead_mask 的形状是 (seq_len, seq_len)
            # 我们需要将 padding_mask 广播到 (seq_len, seq_len)
            # np.minimum 会自动处理广播，确保任何一个掩码为0的位置，最终结果都为0
            self_attention_mask = np.minimum(lookahead_mask, decoder_padding_mask.reshape(1, -1))

        attention_output = self.self_attention.forward(X, X, X, self_attention_mask)
        self.attention_output = self.layer_norm_1.forward(X + attention_output)
        
        # 交叉注意力: 需要传入 encoder_padding_mask
        cross_attention_output = self.cross_attention.forward(self.attention_output, encoder_output, encoder_output, encoder_padding_mask)
        self.cross_attention_output = self.layer_norm_2.forward(self.attention_output + cross_attention_output)
        
        # 前馈网络
        ff_output = self.feed_forward.forward(self.cross_attention_output)
        self.output = self.layer_norm_3.forward(self.cross_attention_output + ff_output)
        
        return self.output
    
    def backward(self, g_prev):

        # 1. 反向传播通过第三个层归一化和残差连接
        g_residual_3 = self.layer_norm_3.backward(g_prev)
        
        # 2. 反向传播通过前馈网络
        g_ff = self.feed_forward.backward(g_residual_3)
        
        # 流向前一个残差块输出的梯度
        g_cross_attention_output = g_ff + g_residual_3

        # 3. 反向传播通过第二个层归一化和残差连接
        g_residual_2 = self.layer_norm_2.backward(g_cross_attention_output)
        
        # 4. 反向传播通过交叉注意力
        # g_Q_cross 是对 self.attention_output 的梯度
        # g_K_cross, g_V_cross 是对 encoder_output 的梯度
        g_Q_cross, g_K_cross, g_V_cross = self.cross_attention.backward(g_residual_2)
        
        # 对 encoder_output 的总梯度
        g_encoder_output = g_K_cross + g_V_cross
        
        # 流向前一个残差块输出 (self.attention_output) 的总梯度
        # 来自交叉注意力的Query路径和第二个残差连接
        g_attention_output = g_Q_cross + g_residual_2

        # 5. 反向传播通过第一个层归一化和残差连接
        g_residual_1 = self.layer_norm_1.backward(g_attention_output)
        
        # 6. 反向传播通过自注意力
        # a, b, c 是对解码器输入 X 的梯度
        a, b, c = self.self_attention.backward(g_residual_1)
        
        # 对解码器输入 X 的总梯度
        # 来自自注意力的Q,K,V路径和第一个残差连接
        g_self_attention = a + b + c + g_residual_1

        return g_self_attention, g_encoder_output


class positionEncoding:
    def __init__(self, d_model, seq_len=5000):
        self.d_model = d_model
        self.seq_len = seq_len
        self.encoding = self._generate_encoding()

    def _generate_encoding(self):
        encoding = np.zeros((self.seq_len, self.d_model))
        position = np.arange(0, self.seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)

        return encoding

    def forward(self, X):
        seq_len = X.shape[0]
        return X + self.encoding[:seq_len]
    
    def backward(self, g_prev):
        return g_prev


class Embedding:
    def __init__(self, vocab_size, d_model, optimizer):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.optimizer = optimizer
        self.embedding_matrix = np.random.randn(vocab_size, d_model) * 0.01  # 初始化嵌入矩阵
        self.scale_factor = np.sqrt(d_model)  # 缩放因子

        self.shared_matrix = False

    def set_shared_embedding(self, shared_embedding):
        """
        设置共享嵌入矩阵
        :param shared_embedding: 共享的嵌入矩阵
        """
        self.shared_matrix = True
        self.embedding_matrix = shared_embedding

    def forward(self, indices):
        """
        前向传播
        :param indices: 输入的索引数组，形状为 (seq_len,)
        :return: 嵌入向量，形状为 (seq_len, d_model)
        """
        self.indices = indices.flatten()  # 确保索引是一维的
        
        # 从嵌入矩阵中查找嵌入向量
        self.output = self.embedding_matrix[self.indices]
        
        # 根据 "Attention Is All You Need" 论文，对嵌入进行缩放
        self.output *= self.scale_factor
        
        return self.output

    def backward(self, g_prev):
        """
        反向传播
        :param g_prev: 来自上一层的梯度，形状为 (seq_len, d_model)
        """
        # 创建一个零梯度的矩阵，形状与嵌入矩阵相同
        g_embedding_matrix = np.zeros_like(self.embedding_matrix)
        
        # 将 g_prev 的梯度累加到 g_embedding_matrix 的相应位置
        # 因为一个词可能在序列中出现多次，所以需要使用 np.add.at
        np.add.at(g_embedding_matrix, self.indices, g_prev * self.scale_factor)

        if self.shared_matrix:
            # 如果是共享嵌入矩阵, 直接返回梯度
            return g_embedding_matrix

        # 使用优化器更新嵌入矩阵的梯度
        self.embedding_matrix -= self.optimizer(self.embedding_matrix, g_embedding_matrix)
        return 0


#交叉熵损失函数
# predictions : 模型预测的概率分布，形状为 (seq_len, vocab_size)
# targets : 目标标签的索引，形状为 (seq_len,)
# 返回的 grad 是 softmax 输入 logits 的梯度，可直接传给线性输出层
def cross_entropy_loss(predictions, targets, ignore_index=0):
    seq_len, vocab_size = predictions.shape
    
    # 创建一个掩码，用于忽略填充标记
    mask = (targets != ignore_index)
    num_valid_tokens = np.sum(mask)
    
    if num_valid_tokens == 0:
        return 0.0, np.zeros_like(predictions)

    # 1. 计算损失
    # 选择对应于目标标签的预测概率
    # 使用 np.arange(seq_len) 来选择每一行的正确索引
    p = predictions[np.arange(seq_len), targets]
    
    # 为避免 log(0)，添加一个很小的数
    log_likelihood = -np.log(p + 1e-9)
    
    # 应用掩码，只计算非填充位置的损失
    loss = np.sum(log_likelihood * mask) / num_valid_tokens
    
    # 2. 计算梯度
    # 梯度是 (p - y)，其中 y 是 one-hot 编码的目标
    grad = predictions.copy()
    grad[np.arange(seq_len), targets] -= 1
    
    # 应用掩码，将填充位置的梯度清零
    grad *= mask[:, np.newaxis]
    
    # 对梯度进行归一化，与损失的计算方式保持一致
    grad /= num_valid_tokens
    
    return loss, grad


def create_pading_mask(seq, pad_token=0):
    seq = np.asarray(seq)
    if seq.ndim == 0:
        raise ValueError("create_pading_mask expects token indices, not a sequence length.")
    return (seq.reshape(-1) != pad_token).astype(np.float32)


def create_padding_mask(seq, pad_token=0):
    return create_pading_mask(seq, pad_token)

def create_look_ahead_mask(size):
    """创建一个用于防止注意力机制看到未来token的下三角掩码。"""
    # 使用 np.tril 创建一个下三角矩阵
    mask = np.tril(np.ones((size, size)))
    return mask.astype('uint8')


class SimpleTransformer:
    def __init__(self):

        self.seq_len = 16
        self.d_model = 32
        self.d_k = 32
        self.d_v = 32
        n_heads = 4
        d_ff = 256

        self.optimizer = nn.AdamWithWarmup(self.d_model, warmup_steps=10000)

        d_model = self.d_model
        d_k = self.d_k
        d_v = self.d_v

        self.wordMap = {
            'P' : 0,
            'S' : 1,
            'E' : 2,
            '1' : 3,
            '2' : 4,
            '3' : 5,
            '4' : 6,
            '5' : 7,
            '6' : 8,
            '7' : 9,
            '8' : 10,
            '9' : 11,
            '0' : 12,
            '+' : 13,
            '=' : 14,
        }

        print(self.wordMap)

        self.input_embedding = Embedding(vocab_size=len(self.wordMap), d_model=d_model, optimizer=self.optimizer)
        self.output_embedding = Embedding(vocab_size=len(self.wordMap), d_model=d_model, optimizer=self.optimizer)

        self.position_encoding = positionEncoding(d_model=d_model, seq_len=self.seq_len)

        # 编码器层
        self.encoder_layers = Encoderlayer(d_model, d_k, d_v, n_heads, d_ff, self.optimizer)
        # 解码器层
        self.decoder_layers = DecoderLayer(d_model, d_k, d_v, n_heads, d_ff, self.optimizer)

        # 输出层
        self.out_linear = nn.Linear(d_model, len(self.wordMap), self.optimizer, with_bias=False)
        self.out_softmax = nn.Softmax()

        # 设置共享嵌入矩阵
        self.input_embedding.set_shared_embedding(self.out_linear.get_shared_weight().T)
        self.output_embedding.set_shared_embedding(self.out_linear.get_shared_weight().T)

    def forward(self, input_indices, target_indices, encoder_padding_mask=None, decoder_padding_mask=None, lookahead_mask=None):
        # 确保输入是 2D
        if input_indices.ndim > 1:
            input_indices = input_indices.reshape(-1)
        if target_indices.ndim > 1:
            target_indices = target_indices.reshape(-1)

        # 编码器前向传播
        input_embeddings = self.input_embedding.forward(input_indices)
        input_with_pos = self.position_encoding.forward(input_embeddings)
        encoder_output = self.encoder_layers.forward(input_with_pos, encoder_padding_mask)

        # 解码器前向传播
        target_embeddings = self.output_embedding.forward(target_indices)
        target_with_pos = self.position_encoding.forward(target_embeddings)
        # 解码器自注意力需要合并 lookahead_mask 和 decoder_padding_mask
        # 交叉注意力需要 encoder_padding_mask (在DecoderLayer内部处理)
        decoder_output = self.decoder_layers.forward(target_with_pos, encoder_output, lookahead_mask, decoder_padding_mask, encoder_padding_mask)

        # 输出层
        output = self.out_linear.forward(decoder_output)
        output = self.out_softmax.forward(output)

        return output
    
    def backward(self, g_prev):
        # 输出层反向传播
        g_decoder_output = self.out_linear.backward(g_prev)

        # 解码器层反向传播
        g_target_indices, g_encoder_output, = self.decoder_layers.backward(g_decoder_output)
        g_target_with_pos = self.position_encoding.backward(g_target_indices)
        g_1 = self.output_embedding.backward(g_target_with_pos)

        # 编码器层反向传播
        g_input_with_pos = self.encoder_layers.backward(g_encoder_output)

        # 位置编码和输入嵌入反向传播
        g_input_embeddings = self.position_encoding.backward(g_input_with_pos)
        g_2 = self.input_embedding.backward(g_input_embeddings)

        self.out_linear.update_shared_weight((g_1 + g_2).T)

    def train(self, input_str, target_str):
        # 1. 准备数据
        input_str = input_str[:self.seq_len - 2]
        target_str = target_str[:self.seq_len - 2]

        encoder_input_str = 'S' + input_str + 'E'
        # 解码器输入是目标序列前加'S'
        decoder_input_str = 'S' + target_str

        # 目标标签是目标序列后加'E'
        target_label_str = target_str + 'E'

        pad_token_idx = self.wordMap['P']

        # 准备编码器输入
        input_indices = [self.wordMap.get(char, pad_token_idx) for char in encoder_input_str]
        input_indices += [pad_token_idx] * (self.seq_len - len(input_indices))
        # 修改这里：直接创建 1D 数组
        input_indices_np = np.array(input_indices)

        # 准备解码器输入
        decoder_input_indices = [self.wordMap.get(char, pad_token_idx) for char in decoder_input_str]
        decoder_input_indices += [pad_token_idx] * (self.seq_len - len(decoder_input_indices))
        # 修改这里：直接创建 1D 数组
        decoder_input_indices_np = np.array(decoder_input_indices)

        # 准备目标标签
        target_label_indices = [self.wordMap.get(char, pad_token_idx) for char in target_label_str]
        target_label_indices += [pad_token_idx] * (self.seq_len - len(target_label_indices))
        target_label_indices_np = np.array(target_label_indices)

        # 2. 创建掩码
        # 掩码的创建现在基于 1D 数组，形状为 (seq_len,)
        encoder_padding_mask = create_padding_mask(input_indices_np, pad_token_idx)
        decoder_padding_mask = create_padding_mask(decoder_input_indices_np, pad_token_idx)
        # 注意：训练时look_ahead_mask仍然需要是完整尺寸
        look_ahead_mask = create_look_ahead_mask(self.seq_len)

        # 3. 前向传播 (一次完成)
        output_probs = self.forward(
            input_indices_np,
            decoder_input_indices_np,
            encoder_padding_mask,
            decoder_padding_mask,
            look_ahead_mask
        )

        # 4. 计算损失
        loss, g_prev = cross_entropy_loss(output_probs, target_label_indices_np, ignore_index=pad_token_idx)

        # 5. 反向传播
        self.backward(g_prev)

        # 6. 更新参数
        self.optimizer.update()

        return loss
    

    def autoregression(self, input_str):
        pad_token_idx = self.wordMap['P']

        # 1. 准备编码器输入
        input_str = input_str[:self.seq_len - 2]
        encoder_input_str = 'S' + input_str + 'E'
        input_indices = [self.wordMap.get(char, self.wordMap['P']) for char in encoder_input_str]

        # 编码器输入仍然需要填充到最大长度
        padded_input_indices = input_indices + [pad_token_idx] * (self.seq_len - len(input_indices))
        # 修改这里：直接创建 1D 数组
        input_indices_np = np.array(padded_input_indices)

        # 创建编码器填充掩码
        # 掩码现在是 1D 的 (seq_len,)
        encoder_padding_mask = create_padding_mask(input_indices_np, pad_token_idx)

        # 2. 编码器前向传播 (只需一次)
        input_embeddings = self.input_embedding.forward(input_indices_np)
        input_with_pos = self.position_encoding.forward(input_embeddings)
        encoder_output = self.encoder_layers.forward(input_with_pos, encoder_padding_mask)

        # 3. 解码器自回归生成
        # 初始化解码器输入，以 'S' 开始，其余用 'P' 填充
        decoder_input_indices = [self.wordMap['S']] + [pad_token_idx] * (self.seq_len - 1)
        
        output_chars = []
        inv_wordMap = {v: k for k, v in self.wordMap.items()}

        for i in range(self.seq_len - 1): # 循环最多 seq_len - 1 次
            # 修改这里：直接创建 1D 数组
            decoder_input_indices_np = np.array(decoder_input_indices)

            # 创建掩码
            look_ahead_mask = create_look_ahead_mask(self.seq_len)
            # 掩码现在是 1D 的 (seq_len,)
            decoder_padding_mask = create_padding_mask(decoder_input_indices_np, pad_token_idx)

            # 解码器前向传播
            target_embeddings = self.output_embedding.forward(decoder_input_indices_np)
            target_with_pos = self.position_encoding.forward(target_embeddings)
            decoder_output = self.decoder_layers.forward(target_with_pos, encoder_output, look_ahead_mask, decoder_padding_mask, encoder_padding_mask)

            # 输出层
            output_logits = self.out_linear.forward(decoder_output)
            output_probs = self.out_softmax.forward(output_logits)

            # 从当前时间步的输出中选择概率最高的词
            predicted_index = np.argmax(output_probs[i, :])
            
            # 如果预测到结束符，则停止
            if predicted_index == self.wordMap['E']:
                break
            
            # 将预测的 token 添加到下一个时间步的输入中
            if i + 1 < self.seq_len:
                decoder_input_indices[i + 1] = predicted_index
            
            output_chars.append(inv_wordMap[predicted_index])

        return "".join(output_chars)


transformer = SimpleTransformer()

wordMap_keys = list(transformer.wordMap.keys())


try:
    nn.set_training_mode(True)
    for epoch in range(50000):
        total_loss = 0
        for i in range(100):
            # 随机生成数字
            input_str = str(np.random.randint(0, 100000)) + '+' + str(np.random.randint(0, 100000))
            target_str = str(eval(input_str))
            #print(f"Input: {input_str}, Target: {target_str}")
            total_loss += transformer.train(input_str, target_str)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / 100:.8f}")
    nn.set_training_mode(False)
except KeyboardInterrupt:
    print("Training interrupted.")

# 测试模型
while True:
    accuracy = 0
    for i in range(1000):
        input_str = str(np.random.randint(0, 100000)) + '+' + str(np.random.randint(0, 100000))
        target_str = str(eval(input_str))
        predicted_output = transformer.autoregression(input_str)
        if target_str == predicted_output:
            accuracy += 1
    print(f"Accuracy over 1000 samples: {accuracy / 10:.2f}%")

    # test_input = input("Enter a string:")
    # predicted_output = transformer.autoregression(test_input)
    # print(f"Input: {test_input}, Predicted Output: {predicted_output}")
