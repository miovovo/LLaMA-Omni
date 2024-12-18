import torch
import torch.nn as nn
from torch import Tensor
from conformer.conformer.encoder import ConformerEncoder
from speech_projector.builder import build_speech_projector

# 测试代码
def test_conformer_encoder():
    batch_size = 3
    seq_len = 12345
    feature_dim = 80

    # 创建随机输入数据
    inputs = torch.randn(batch_size, seq_len, feature_dim)
    input_lengths = torch.tensor([seq_len, seq_len - 100, seq_len - 200])  # 假设每个样本有不同长度

    # 初始化 ConformerEncoder
    model = ConformerEncoder(
        input_dim=feature_dim,
        encoder_dim=512,
        num_layers=5,
        num_attention_heads=8,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        input_dropout_p=0.1,
        feed_forward_dropout_p=0.1,
        attention_dropout_p=0.1,
        conv_dropout_p=0.1,
        conv_kernel_size=31,
        half_step_residual=True
    )

    # 打印模型参数数量
    print(f"Model parameters: {model.count_parameters()}")

    # 前向传播
    outputs, output_lengths = model(inputs, input_lengths)

    # 验证输出形状
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Input lengths: {input_lengths}")
    print(f"Output lengths: {output_lengths}")

# 运行测试
test_conformer_encoder()
