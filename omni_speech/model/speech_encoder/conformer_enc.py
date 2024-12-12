import torch
import torch.nn as nn

class ConformerEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 128,  # 输入特征维度
            encoder_dim: int = 1280,  # 编码器内部维度
            num_layers: int = 17,  # Conformer 层数
            num_attention_heads: int = 8,  # 多头注意力头数
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerEncoder, self).__init__()

        # 输入投影层，将输入维度映射到 encoder_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )

        # 定义 Conformer Block
        self.layers = nn.ModuleList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual,
            ) for _ in range(num_layers)
        ])

        # 输出投影层
        self.output_projection = nn.Linear(encoder_dim, encoder_dim)

        # Downsample 操作调整序列长度
        self.downsample = nn.Conv1d(encoder_dim, encoder_dim, kernel_size=2, stride=2)

    def forward(self, x):
        # x: [batch_size, feature_dim, seq_len]

        # 将输入从 [batch_size, feature_dim, seq_len] 转为 [batch_size, seq_len, feature_dim]
        x = x.permute(0, 2, 1)

        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, encoder_dim]

        # Conformer Block 编码
        for layer in self.layers:
            x = layer(x)  # [batch_size, seq_len, encoder_dim]

        # Downsample 调整序列长度，从 3000 -> 1500
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, encoder_dim, seq_len]
        x = self.downsample(x)  # [batch_size, encoder_dim, seq_len // 2]
        x = x.permute(0, 2, 1)  # 转回 [batch_size, seq_len, encoder_dim]

        # 输出投影
        x = self.output_projection(x)  # [batch_size, seq_len, encoder_dim]

        return x

class ConformerBlock(nn.Module):
    def __init__(self,
                 encoder_dim: int,
                 num_attention_heads: int,
                 feed_forward_expansion_factor: int,
                 conv_expansion_factor: int,
                 feed_forward_dropout_p: float,
                 attention_dropout_p: float,
                 conv_dropout_p: float,
                 conv_kernel_size: int,
                 half_step_residual: bool):
        super(ConformerBlock, self).__init__()
        # 这里是 ConformerBlock 的实现，可以参考之前的内容
        pass  # 为简洁省略具体实现

# 测试模型
batch_size = 1
input_dim = 128
seq_len = 3000
encoder_dim = 1280

model = ConformerEncoder(input_dim=input_dim, encoder_dim=encoder_dim)
input_tensor = torch.randn(batch_size, input_dim, seq_len)  # [1, 128, 3000]
output = model(input_tensor)  # [1, 1500, 1280]
print("Output shape:", output.shape)
