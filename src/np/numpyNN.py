import numpy as np


# Global variable to control training mode
training_mode = False
def set_training_mode(mode):
    global training_mode
    training_mode = mode

#标准梯度下降优化器
class SGDopt:
    def __init__(self, lr_max, lr_min=-1, decay_steps=50, decay_rate=0.99):
        """
        初始化梯度下降优化器
        :param lr_max: 初始学习率
        :param lr_min: 最小学习率
        :param decay_steps: 学习率衰减步数
        """
        self.lr_max = lr_max
        self.lr_min = lr_min if lr_min >= 0 else lr_max
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.step_count = 0
        # 初始化学习率为最大值
        self.lr = lr_max

    def update(self):
        # 更新学习率
        if self.step_count > self.decay_steps:
            self.lr = max(self.lr_min, self.lr * self.decay_rate)
            self.step_count = 0
        self.step_count += 1

    def __call__(self, params, grads):
        # 执行参数更新
        return self.lr * grads

class AdaptiveGDopt:
    def __init__(self, lr=0.01, epsilon=1e-8):
        """
        初始化自适应梯度下降优化器 (Adagrad)
        :param lr: 学习率
        :param epsilon: 防止除以零的小常数，用于数值稳定性
        """
        self.lr = lr
        self.epsilon = epsilon
        self.cache = {}  # 使用字典来存储每个参数的缓存

    def update(self):
        """
        Adagrad 不需要像标准梯度下降那样显式地进行学习率衰减，
        因为学习率会根据累积的梯度自动调整。
        """
        pass

    def __call__(self, params, grads):
        """
        使用 Adagrad 算法计算参数的更新量。
        :param params: 当前参数 (用于查找缓存)
        :param grads: 当前参数的梯度
        :return: 参数的更新量 (delta)
        """
        param_id = id(params)
        if param_id not in self.cache:
            self.cache[param_id] = np.zeros_like(params)
        
        # 累积梯度的平方
        self.cache[param_id] += grads**2

        # 计算参数更新量
        # 学习率会根据 cache 的大小进行调整
        return self.lr * grads / (np.sqrt(self.cache[param_id]) + self.epsilon)


class AdamWithWarmup:
    def __init__(self, d_model, warmup_steps, beta1=0.9, beta2=0.98, epsilon=1e-9):
        """
        Adam优化器，带有"Attention Is All You Need"中描述的学习率调度。
        :param d_model: 模型的维度。
        :param warmup_steps: 预热步数。
        :param beta1: Adam的beta1参数。
        :param beta2: Adam的beta2参数。
        :param epsilon: 防止除以零的小常数。
        """
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.step_num = 0
        self.m = {}  # 梯度的一阶矩估计
        self.v = {}  # 梯度的二阶矩估计

    def update(self):
        """
        在每个训练步骤结束时调用，以增加步数计数器。
        """
        self.step_num += 1

    def __call__(self, params, grads):
        """
        为给定的参数和梯度计算更新量。
        :param params: 模型参数 (用于获取唯一ID以存储状态)。
        :param grads: 参数的梯度。
        :return: 参数的更新量 (delta)。
        """
        # 确保步数从1开始，以避免在偏差校正中除以零
        current_step = self.step_num + 1
        
        # 1. 计算当前学习率
        arg1 = current_step ** -0.5
        arg2 = current_step * (self.warmup_steps ** -1.5)
        lr = (self.d_model ** -0.5) * min(arg1, arg2)

        # 2. Adam算法更新
        param_id = id(params)
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(params)
            self.v[param_id] = np.zeros_like(params)

        # 更新一阶和二阶矩估计
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grads
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grads ** 2)

        # 计算偏差校正后的矩估计
        m_hat = self.m[param_id] / (1 - self.beta1 ** current_step)
        v_hat = self.v[param_id] / (1 - self.beta2 ** current_step)

        # 计算参数更新量
        delta = lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return delta

class Linear:
    def __init__(self, input_cols, output_cols, optimizer, with_bias = False):
        self.with_bias = with_bias
        self.opt = optimizer
        self.Y = np.zeros((1, output_cols))
        self.X = np.zeros((1, input_cols))
        self.G = np.zeros((1, input_cols))
        self.W = np.random.randn(input_cols, output_cols) * 0.1

        self.shared_weight = False
        if with_bias:
            self.B = np.random.randn(1, output_cols) * 0.01
        else:
            self.B = None

    def get_shared_weight(self):
        """
        获取共享权重
        :return: 权重矩阵W
        """
        self.shared_weight = True
        return self.W
    
    def forward(self, X):
        # 在推理模式下，直接计算输出
        self.X = X
        self.Y = np.matmul(X, self.W)
        if self.with_bias:
            self.Y += self.B
        return self.Y

    def backward(self, prev_G):
        # 计算梯度
        self.G = np.matmul(self.X.T, prev_G)
        if self.with_bias:
            self.B_grad = np.sum(prev_G, axis=0, keepdims=True)
        else:
            self.B_grad = None

        # 反向传播到上一层必须使用前向传播时的权重
        input_grad = np.matmul(prev_G, self.W.T)

        if self.shared_weight:
            # 如果是共享权重，直接返回梯度
            return input_grad

        # 更新权重
        self.W -= self.opt(self.W, self.G)
        if self.with_bias:
            self.B -= self.opt(self.B, self.B_grad)

        # 返回上一层的梯度
        return input_grad
    
    def update_shared_weight(self, total_g):
        """
        更新共享权重
        :param new_weight: 新的权重矩阵
        """
        total_g = total_g + self.G # 累加梯度
        if self.shared_weight:
            self.W -= self.opt(self.W, total_g)
            if self.with_bias and self.B_grad is not None:
                self.B -= self.opt(self.B, self.B_grad)
        else:
            raise ValueError("This layer does not have shared weights.")


class DropOut:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.Y = np.matrix([])
        self.X = np.matrix([])

    def forward(self, X):
        self.X = X
        if self.dropout_rate > 0:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=X.shape) / (1 - self.dropout_rate)
            self.Y = X * self.mask
        else:
            self.Y = X
        return self.Y

    def backward(self, prev_G):
        if self.dropout_rate > 0:
            return prev_G * self.mask
        else:
            return prev_G


class ResidualConnection:
    def __init__(self, sublayer):
        self.sublayer = sublayer
        self.Y = np.matrix([])
        self.X = np.matrix([])

    def forward(self, X):
        self.X = X
        self.Y = X + self.sublayer.forward(X)
        return self.Y
    
    def backward(self, prev_G):
        grad_from_sublayer = self.sublayer.backward(prev_G)
        self.G = grad_from_sublayer + prev_G
        return self.G

class Softmax:
    Y = np.matrix([]) # output
    X = np.matrix([]) # input
    G = np.matrix([]) # gradient

    def forward(self, X):
        self.X = X
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # for numerical stability
        self.Y = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.Y

    def backward(self, prev_G):
        self.G = self.Y * (prev_G - np.sum(prev_G * self.Y, axis=1, keepdims=True))
        return self.G

class ReLU:
    Y = np.matrix([]) # output
    X = np.matrix([]) # input
    G = np.matrix([]) # gradient

    def forward(self, X):
        self.X = X
        self.Y = np.maximum(0, X)
        return self.Y

    def backward(self, prev_G):
        self.G = prev_G * (self.X > 0)
        return self.G
    
class FeedForward:
    def __init__(self, d_model, d_ff, optimizer, dropout_rate=0.0):
        self.lr = optimizer
        self.linear1 = Linear(d_model, d_ff, optimizer)
        self.dropout = DropOut(dropout_rate) if dropout_rate > 0 else None
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model, optimizer)

    def forward(self, X):
        t = self.linear1.forward(X)
        if self.dropout:
            t = self.dropout.forward(t)
        t = self.relu.forward(t)
        self.Y = self.linear2.forward(t)
        return self.Y

    def backward(self, prev_G):
        grad_from_linear2 = self.linear2.backward(prev_G)
        grad_from_relu = self.relu.backward(grad_from_linear2)
        if self.dropout:
            grad_from_relu = self.dropout.backward(grad_from_relu)
        # 反向传播到第一个线性层
        self.G = self.linear1.backward(grad_from_relu)
        return self.G

class LayerNorm:
    def __init__(self, d_model, optimizer, eps=1e-8, ):
        """
        层归一化
        Args:
            d_model: 特征维度
            eps: 数值稳定性参数
        """
        self.d_model = d_model
        self.eps = eps
        self.opt = optimizer
        # 可学习参数
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, X):
        self.X = X
        # 计算均值和方差（在最后一个维度上）
        self.mean = np.mean(X, axis=-1, keepdims=True)
        self.var = np.var(X, axis=-1, keepdims=True)

        # 标准化
        self.X_norm = (X - self.mean) / np.sqrt(self.var + self.eps)
        
        # 缩放和平移
        self.output = self.gamma * self.X_norm + self.beta
        return self.output
    
    def backward(self, g_prev):
        # g_prev shape: (seq_len, d_model)
        
        # 对gamma和beta的梯度
        g_gamma = np.sum(g_prev * self.X_norm, axis=0)
        g_beta = np.sum(g_prev, axis=0)

        # 对输入的梯度
        g_X_norm = g_prev * self.gamma
        
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        g_var = np.sum(g_X_norm * (self.X - self.mean) * (-0.5) * std_inv**3, axis=-1, keepdims=True)
        g_mean = np.sum(g_X_norm * (-std_inv), axis=-1, keepdims=True) + \
                 g_var * np.mean(-2.0 * (self.X - self.mean), axis=-1, keepdims=True)
        
        # 反向传播到 X
        # shape: (seq_len, d_model)
        g_X = g_X_norm * std_inv + g_var * (2.0 * (self.X - self.mean) / self.d_model) + g_mean / self.d_model

        # 输入梯度计算完成后再更新参数，避免使用到更新后的 gamma / beta
        self.gamma -= self.opt(self.gamma, g_gamma)
        self.beta -= self.opt(self.beta, g_beta)
        
        return g_X

class Layers:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, prev_G):
        for layer in reversed(self.layers):
            prev_G = layer.backward(prev_G)
        return prev_G


# Absolute value loss
# return loss, gradient
# Y: true labels, Y_hat: predicted labels
def absolute_loss(Y, Y_hat):
    loss = np.sum(np.abs(Y - Y_hat))
    G = Y_hat - Y  # Gradient of absolute loss
    return loss, G

