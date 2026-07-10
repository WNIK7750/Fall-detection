"""
=============================================================================
摔倒检测系统 - 增强模型架构
阶段1优化：BiLSTM + BiGRU + 双向Attention

架构说明：
1. BiLSTM：双向建模，能同时利用"摔倒前"和"摔倒后"的信息
2. BiGRU：补充LSTM，GRU对短序列更友好
3. 双向Attention：两个方向的attention权重，捕捉关键摔倒时刻
4. 特征拼接：多尺度特征融合

论文参考：BiLSTM-BiGRU-Attention (SOTA, 99.5% Acc)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalAttention(nn.Module):
    """
    双向自注意力机制
    同时计算前向和后向的注意力权重，聚焦关键时刻
    """

    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # 多头注意力
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        # 前向/后向门控（控制信息流向）
        self.forward_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.backward_gate = nn.Linear(hidden_size * 2, hidden_size)

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        x: [batch, seq_len, hidden_size]
        返回: [batch, seq_len, hidden_size], [batch, seq_len] 注意力权重
        """
        batch, seq_len, _ = x.shape

        # === 多头自注意力 ===
        Q = self.query(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)

        # 输出投影 + 残差连接
        attn_output = self.out(attn_output)
        x = self.layer_norm(x + self.dropout(attn_output))

        # === 双向门控（区分前向/后向信息） ===
        forward_h = x[:, :-1, :]      # [batch, seq_len-1, hidden]
        backward_h = x[:, 1:, :].flip(1)  # [batch, seq_len-1, hidden] 反向

        # 拼接前向和后向状态
        combined = torch.cat([forward_h, backward_h], dim=-1)

        # 门控机制
        f_gate = torch.sigmoid(self.forward_gate(combined))
        b_gate = torch.sigmoid(self.backward_gate(combined))

        # 应用门控
        forward_out = f_gate * forward_h
        backward_out = b_gate * backward_h

        # 合并双向输出
        # 前 seq_len-1 位置用门控前向
        bidirectional = torch.cat([forward_out, backward_out[:, -1:, :]], dim=1)

        # 注意力权重（用于可视化/分析）
        avg_attn = attn_weights.mean(dim=1)  # [batch, seq_len, seq_len] -> [batch, seq_len, seq_len]
        attn_agg = avg_attn.sum(dim=2)  # [batch, seq_len]

        return bidirectional, attn_agg


class EnhancedFallDetector(nn.Module):
    """
    增强摔倒检测模型：BiLSTM + BiGRU + 双向Attention

    输入: [batch, seq_len, 10或20]  10维或20维特征
    输出: [batch, 2]  二分类 logits
    """

    def __init__(self, input_size=10, hidden_size=128, num_classes=2,
                 dropout=0.3, num_heads=4):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # === 输入投影层 ===
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # === BiLSTM 层 ===
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # === BiGRU 层 ===
        self.bigru = nn.GRU(
            input_size=hidden_size * 2,  # BiLSTM output
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        # === 双向Attention层 ===
        self.attention = BidirectionalAttention(
            hidden_size=hidden_size * 2,  # BiGRU output
            num_heads=num_heads
        )

        # === 时序池化层 ===
        self.temporal_pool = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # === 分类头 ===
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, return_attention=False):
        """
        前向传播

        x: [batch, seq_len, input_size]
        return_attention: 是否返回注意力权重（用于可视化）
        """
        # === 1. 输入投影 ===
        x = self.input_proj(x)  # [batch, seq_len, hidden_size]

        # === 2. BiLSTM ===
        lstm_out, _ = self.bilstm(x)  # [batch, seq_len, hidden_size * 2]
        lstm_out = F.dropout(lstm_out, p=0.2, training=self.training)

        # === 3. BiGRU ===
        gru_out, _ = self.bigru(lstm_out)  # [batch, seq_len, hidden_size * 2]
        gru_out = F.dropout(gru_out, p=0.2, training=self.training)

        # === 4. 双向Attention ===
        attn_out, attn_weights = self.attention(gru_out)

        # === 5. 加权池化（替代简单的last state） ===
        # 用attention权重的归一化版本作为池化权重
        attn_norm = F.softmax(attn_weights, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
        pooled = (attn_out * attn_norm).sum(dim=1)  # [batch, hidden_size * 2]

        # === 6. 分类 ===
        pooled = self.temporal_pool(pooled)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attn_weights
        return logits


class CompactEnhancedDetector(nn.Module):
    """
    紧凑版增强模型（参数更少，适合小数据集）
    BiLSTM + 简单Attention
    """

    def __init__(self, input_size=20, hidden_size=64, num_classes=2, dropout=0.3):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 简单单头Attention
        self.attention_weights = nn.Linear(hidden_size * 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # BiLSTM
        lstm_out, _ = self.bilstm(x)

        # 注意力权重
        attn_scores = self.attention_weights(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)

        # 加权求和
        context = (lstm_out * attn_weights).sum(dim=1)

        return self.classifier(context)


# ============ 模型工厂 ============
def build_enhanced_model(input_size=10, model_type='full', **kwargs):
    """
    构建增强模型

    model_type: 'full' -> 完整架构(BiLSTM+BiGRU+Attention)
                'compact' -> 紧凑架构(BiLSTM+Attention)
    """
    if model_type == 'full':
        return EnhancedFallDetector(input_size=input_size, **kwargs)
    elif model_type == 'compact':
        return CompactEnhancedDetector(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 快速测试
    print("测试增强模型架构...")

    # 全量模型（10维特征）
    model_full = build_enhanced_model(input_size=10, model_type='full')
    x = torch.randn(4, 20, 10)  # [batch=4, seq=20, feat=10]
    out = model_full(x)
    print(f"Full模型输出: {out.shape}")  # [4, 2]
    print(f"Full模型参数量: {sum(p.numel() for p in model_full.parameters()):,}")

    # 紧凑模型（10维特征）
    model_compact = build_enhanced_model(input_size=10, model_type='compact')
    out = model_compact(x)
    print(f"Compact模型输出: {out.shape}")
    print(f"Compact模型参数量: {sum(p.numel() for p in model_compact.parameters()):,}")

    # 全量模型（20维特征）
    model_full_20 = build_enhanced_model(input_size=20, model_type='full')
    x20 = torch.randn(4, 20, 20)  # [batch=4, seq=20, feat=20]
    out20 = model_full_20(x20)
    print(f"Full模型(20维)输出: {out20.shape}")
    print(f"Full模型(20维)参数量: {sum(p.numel() for p in model_full_20.parameters()):,}")

    print("模型测试通过!")
