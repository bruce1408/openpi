#!/usr/bin/env python3
r"""依据 OpenPI 官方数据流编写的、可直接运行的极简 :math:`\pi_{0.5}`。

这个文件的目的是讲清算法，而不是加载官方 checkpoint 或复现官方精度。它保留了
π₀.₅-LIBERO 的图像、文本、动作等外部张量尺寸，同时用很小的网络替代内部大模型：

==========================  ================================================
官方 OpenPI                 本文件中的教学替代实现
==========================  ================================================
PaliGemma SentencePiece     确定性的 UTF-8 Byte Tokenizer
SigLIP Vision Transformer   Patch Projection + 2 层 Tiny Transformer
Gemma-2B（18 层）           2 层 Prefix Transformer
Gemma-300M Action Expert    2 层 Tiny Action Expert
逐层 AdaRMS                 保留 scale、shift、gate 三路时间调制
Flow Matching               保留相同的 x_t、目标速度与 MSE
10 步 Euler Solver          保留相同的 noise -> action 积分方向
==========================  ================================================

本文件对应公开 OpenPI 中 ``pi05_libero``/NPU 路径使用的低层连续动作生成部分，
有意省略论文中的高层语言自回归循环和 FAST 离散动作预训练。

阅读时要区分三种“时间”：

1. 控制时间：机器人在真实环境中的第几个控制周期；
2. 动作块时间：一次预测的 H 个未来动作，这里 H=10；
3. Flow 时间 t：把噪声变成动作的人工时间，代码约定 t=1 是噪声、t=0 是动作。

推理时，图像和文本 Prefix 只计算一次，并缓存每一层的 K/V；十次 Euler 更新只重复
运行较小的 Action Expert。这正是 Prefix K/V Cache 的主要价值。

快速运行（主要检查张量形状和数据流）：

    python examples/minimal_pi05.py \
        --train-steps 2 --flow-steps 10

增加训练步数，便于观察 loss 下降：

    python examples/minimal_pi05.py \
        --train-steps 20 --flow-steps 10
"""


import argparse
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TinyPi05Config:
    """集中定义尺寸：输入/输出契约对齐 pi05_libero，内部网络保持 Tiny。"""

    # ---------- 与官方 pi05_libero 对齐的外部尺寸 ----------
    image_size: int = 224       # 官方预处理统一 Resize 到 224×224
    patch_size: int = 14        # 官方 SigLIP 变体为 So400m/14，即每块 patch 是 14×14 像素
    num_image_views: int = 3    # base、left_wrist、right_wrist 三个固定图像槽
    max_text_tokens: int = 200  # Pi0Config(pi05=True) 的默认文本长度
    action_horizon: int = 10    # pi05_libero 明确覆盖为一次预测 10 个未来动作
    action_dim: int = 32        # OpenPI 把不同机器人的动作统一 pad 到 32 维
    libero_action_dim: int = 7  # LIBERO 原始有效动作维度；其余 25 维只是 padding

    # ---------- 用于替代官方大模型的 Tiny 内部尺寸 ----------
    vision_depth: int = 2      # 官方 SigLIP So400m 是 27 层；这里仅保留 2 层
    
    # 官方 PaliGemma 词表是 257,152；这里改成 4 个特殊 ID + 256 种 UTF-8 byte 值。
    vocab_size: int = 260
    width: int = 32            # 官方 Gemma-2B/Expert 宽度为 2048/1024；这里统一缩到 32
    num_heads: int = 4         # 教学版每个 head_dim=32/4=8
    mlp_dim: int = 64          # 官方 Gemma-2B/Expert MLP 为 16384/4096；这里缩到 64
    prefix_depth: int = 2      # 用 2 层代替 Gemma-2B 的 18 层
    expert_depth: int = 2      # 用 2 层代替 Action Expert 的 18 层

    def __post_init__(self) -> None:
        # Patch 划分必须整除图像，否则最后一行/列像素无法形成完整 patch。
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        
        # Multi-Head Attention 要把 width 均匀拆成 num_heads 份。
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")
        
        if self.libero_action_dim > self.action_dim:
            raise ValueError("libero_action_dim cannot exceed the padded model action_dim")

        # OpenPI 中 Expert 第 i 层恰好读取 Prefix 第 i 层缓存的 K/V。
        if self.prefix_depth != self.expert_depth:
            raise ValueError("prefix_depth and expert_depth must match")

    @property
    def patch_grid_size(self) -> int:
        """单边 patch 数量：224 / 14 = 16。"""

        return self.image_size // self.patch_size

    @property
    def image_tokens_per_view(self) -> int:
        """单个相机视角的视觉 token 数：16 × 16 = 256。"""

        return self.patch_grid_size**2

    @property
    def max_prefix_tokens(self) -> int:
        """全部图像槽和定长文本组成的最大 Prefix 长度：3 × 256 + 200 = 968。"""

        return self.num_image_views * self.image_tokens_per_view + self.max_text_tokens


class TinyTextTokenizer:
    """PaliGemma SentencePiece 的无依赖教学替代品。

    官方实现加载 ``paligemma_tokenizer.model``，通过 SentencePiece 把子词映射为整数 ID。
    为了让本文件离线即可运行，这里把每个 UTF-8 byte 直接映射为一个 ID。它的语言学
    效果不等价于 SentencePiece，但完整保留了下面这个接口：

        字符串 -> token IDs [B, T] + 有效位置 mask [B, T]
    """

    PAD = 0          # Padding：补齐长度的位置，后续 Attention 必须屏蔽
    BOS = 1          # Beginning Of Sequence：句子开始
    EOS = 2          # End Of Sequence：句子结束
    BYTE_OFFSET = 4  # 0~3 留给特殊符号，因此 byte 0 映射为 token 4
    VOCAB_SIZE = 256 + BYTE_OFFSET

    def __init__(self, max_length: int = 200):
        self.max_length = max_length

    def encode(self, prompt: str) -> tuple[list[int], list[bool]]:
        """编码单条指令，返回定长 token ID 和与之对应的 bool mask。"""

        # 与 OpenPI 类似：清理首尾空格，把下划线/换行统一为空格；结尾换行表示回答起点。
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ") + "\n"

        # 文本主体的每个 UTF-8 byte 都落在 [4, 259]，前后加 BOS/EOS。
        token_ids = [self.BOS]
        token_ids.extend(byte + self.BYTE_OFFSET for byte in cleaned.encode("utf-8"))
        token_ids.append(self.EOS)

        # 超长时截断，并强制最后一个位置仍然是 EOS，避免得到没有结束标记的句子。
        token_ids = token_ids[: self.max_length]
        if len(token_ids) == self.max_length:
            token_ids[-1] = self.EOS

        # mask=True 表示真实 token，False 表示 padding；Attention 会屏蔽 False 的 Key。
        valid = [True] * len(token_ids)
        padding = self.max_length - len(token_ids)
        token_ids.extend([self.PAD] * padding)
        valid.extend([False] * padding)
        return token_ids, valid

    def batch_encode(self, prompts: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
        """把多条字符串堆成 [B, T]，并直接放到模型所在 device。"""

        encoded = [self.encode(prompt) for prompt in prompts]
        ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
        mask = torch.tensor([item[1] for item in encoded], dtype=torch.bool, device=device)
        return ids, mask


class RMSNorm(nn.Module):
    r"""Gemma 风格的 RMSNorm。

    与 LayerNorm 不同，RMSNorm 不减均值，只用均方根归一化：

        y = x / sqrt(mean(x²) + eps) * (1 + scale)

    ``scale`` 从 0 初始化，所以初始缩放系数是 1，而不是 0。
    """

    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(width))

    def forward(self, x: Tensor) -> Tensor:
        # 归一化统计使用 FP32，避免 BF16/FP16 平方与归约造成较大数值误差。
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return normalized * (1.0 + self.scale)


class AdaRMSNorm(nn.Module):
    r"""由 Flow 时间调制的 Adaptive RMSNorm，对应 π0.5 Action Expert 的 AdaRMS。

    普通 RMSNorm 的参数对所有样本固定；AdaRMS 根据当前 Flow 时间生成三组参数：

        normalized = RMS(x)
        scale, shift, gate = Linear(time_condition)
        y = normalized * (1 + scale) + shift
        residual = x + gate * block(y)

    因而网络在 t≈1（接近噪声）和 t≈0（接近动作）时可以采用不同处理策略。
    modulation 零初始化会令 scale=shift=gate=0，使每个残差块初始近似恒等映射，
    训练开始时更稳定。
    """

    def __init__(self, width: int, condition_width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 一次线性映射同时产生三份 [B, width] 参数。
        self.modulation = nn.Linear(condition_width, 3 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        # x: [B, S, D]；condition: [B, D]。
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        
        # unsqueeze(1) 后是 [B, 1, 3D]，可广播到序列长度 S。
        scale, shift, gate = self.modulation(condition).unsqueeze(1).chunk(3, dim=-1)
        return normalized * (1.0 + scale) + shift, gate


class GemmaMLP(nn.Module):
    r"""保留 Gemma Gate/Up/Down 结构的门控 MLP。

    计算形式是：

        output = down(GELU(gate(x)) * up(x))

    逐元素相乘让 ``gate`` 分支控制哪些中间特征可以通过。
    """

    def __init__(self, width: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden_dim, bias=False)
        self.up_proj = nn.Linear(width, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # [B, S, D] -> 两个 [B, S, mlp_dim] -> [B, S, D]
        gated = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        return self.down_proj(gated)


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    """[B, S, D] -> [B, heads, S, head_dim]，为每个注意力头分出独立维度。"""

    batch, sequence, width = x.shape
    head_dim = width // num_heads
    return x.view(batch, sequence, num_heads, head_dim).transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    """_split_heads 的逆操作：[B, heads, S, head_dim] -> [B, S, D]。"""

    batch, heads, sequence, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, sequence, heads * head_dim)


def _attention(q: Tensor, k: Tensor, v: Tensor, key_valid: Tensor) -> Tensor:
    r"""显式实现 Scaled Dot-Product Attention，便于观察 Prefix K/V 如何复用。

    输入形状：
      q: [B, heads, query_len, head_dim]
      k/v: [B, heads, key_len, head_dim]
      key_valid: [B, key_len]

    核心公式：softmax(QKᵀ / sqrt(head_dim))V。
    """

    # 除以 sqrt(head_dim)，防止 head_dim 增大时点积值过大、softmax 过度饱和。
    scale = q.shape[-1] ** -0.5
    
    # 注意力 logits 使用 FP32 计算，提高数值稳定性。
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    
    # padding Key 设为极小值，softmax 后概率约为 0，因此不会影响有效 token。
    logits = logits.masked_fill(~key_valid[:, None, None, :], torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=-1).to(q.dtype)
    return torch.matmul(probabilities, v)


class MultiHeadAttention(nn.Module):
    """教学版多头注意力，用普通 MHA 简化官方 Gemma 的 MQA/GQA。"""

    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, key_valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Prefix 自注意力：返回层输出，同时暴露本层 K/V 供 Action Expert 缓存。"""

        # 四个投影都保持 [B, S, D]；split 后得到 [B, heads, S, head_dim]。
        q = _split_heads(self.q_proj(x), self.num_heads)
        k = _split_heads(self.k_proj(x), self.num_heads)
        v = _split_heads(self.v_proj(x), self.num_heads)
        output = self.out_proj(_merge_heads(_attention(q, k, v, key_valid)))
        return output, k, v

    def forward_with_prefix(
        self,
        x: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
        prefix_valid: Tensor,
    ) -> Tensor:
        """让动作 Query 同时关注缓存的图文 K/V 与当前动作 K/V。

        这一步对应官方 Denoise 阶段的 ``past_key_values``：

            K_total = concat(K_prefix, K_action)
            V_total = concat(V_prefix, V_action)

        Prefix 不随 Euler 步变化，所以不需要重新计算；x_t 与 t 每步变化，因此动作
        Q/K/V 必须每步重新计算。
        """

        # Action Expert 使用自己的 q/k/v 权重，不与 Prefix Gemma 共享权重。
        q = _split_heads(self.q_proj(x), self.num_heads)
        action_k = _split_heads(self.k_proj(x), self.num_heads)
        action_v = _split_heads(self.v_proj(x), self.num_heads)
        
        # 序列维位于倒数第二维，所以在 dim=-2 拼接 Prefix 和 Action。
        k = torch.cat([prefix_k, action_k], dim=-2)
        v = torch.cat([prefix_v, action_v], dim=-2)
        
        # 所有 action token 都是真实 token；Prefix 的 padding 状态由 prefix_valid 给出。
        action_valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        key_valid = torch.cat([prefix_valid, action_valid], dim=-1)
        return self.out_proj(_merge_heads(_attention(q, k, v, key_valid)))


class PrefixBlock(nn.Module):
    """一层双向 Tiny Gemma Prefix Block，并把本层 K/V 显式暴露出来。

    Prefix 内的图像和文本 token 使用双向注意力，可以彼此完整交互。官方模型为避免
    重复处理不变的观察条件，会缓存每一层注意力计算出的 K/V。
    """

    def __init__(self, width: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.attention_norm = RMSNorm(width)
        self.attention = MultiHeadAttention(width, num_heads)
        self.mlp_norm = RMSNorm(width)
        self.mlp = GemmaMLP(width, mlp_dim)

    def forward(self, x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Pre-Norm Attention：先归一化，再注意力，最后与原输入做残差相加。
        attention_input = self.attention_norm(x)
        attention_output, k, v = self.attention(attention_input, valid)
        x = x + attention_output
        # Pre-Norm MLP：与 Gemma 的第二个残差分支对应。
        x = x + self.mlp(self.mlp_norm(x))
        # x 传给 Prefix 下一层；k/v 留给同层编号的 Action Expert。
        return x, k, v


class ActionExpertBlock(nn.Module):
    """一层 Action Expert：独立 Transformer 权重 + Flow 时间条件 AdaRMS。

    官方实现可以理解成同一层里存在两套专家权重：图像/文本走 PaliGemma 权重，
    连续动作 token 走较小的 Action Expert 权重。两套权重通过 Attention 的 K/V 交互。
    本类只实现动作专家这一侧。
    """

    def __init__(self, width: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.attention_norm = AdaRMSNorm(width, width)
        self.attention = MultiHeadAttention(width, num_heads)
        self.mlp_norm = AdaRMSNorm(width, width)
        self.mlp = GemmaMLP(width, mlp_dim)

    def forward(
        self,
        x: Tensor,
        time_condition: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
        prefix_valid: Tensor,
    ) -> Tensor:
        
        # 第一条残差支路：AdaRMS -> 对 Prefix/Action 联合 K/V 做 Attention -> Gate -> Add。
        attention_input, attention_gate = self.attention_norm(x, time_condition)
        attention_output = self.attention.forward_with_prefix(
            attention_input, prefix_k, prefix_v, prefix_valid
        )
        x = x + attention_gate * attention_output

        # 第二条残差支路：AdaRMS -> Gemma MLP -> Gate -> Add。
        mlp_input, mlp_gate = self.mlp_norm(x, time_condition)
        x = x + mlp_gate * self.mlp(mlp_input)
        return x


class TinyVisionEncoder(nn.Module):
    """SigLIP 的教学替代：Patch Embedding + 位置编码 + 2 层 Transformer。

    它只模拟 SigLIP 在 π0.5 中承担的主要职责：

        单个视角 [B, 3, 224, 224] -> 视觉 token [B, 256, width]

    它没有复现 SigLIP 的真实预训练、网络深度和投影权重，因此不具备真实视觉能力。

    这里必须特别区分：``patch_size=14`` 表示每个 patch 覆盖 14×14 像素，而不是
    “总共有 14×14 个 patch”。因为 224/14=16，所以每个视角产生 16×16=256 个 token。
    """

    def __init__(self, config: TinyPi05Config):
        super().__init__()
        
        # kernel=stride=patch_size，相当于把互不重叠的 patch 直接投影到 width 维。
        self.patch_projection = nn.Conv2d(
            3, config.width, kernel_size=config.patch_size, stride=config.patch_size
        )
        num_patches = config.image_tokens_per_view # 256
        
        # Transformer 本身不知道 patch 的二维位置，因此要为每个位置加入可学习编码。
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, config.width)) # 1,256,32
        nn.init.normal_(self.position_embedding, std=0.02)
        self.layers = nn.ModuleList(
            [
                PrefixBlock(config.width, config.num_heads, config.mlp_dim)
                for _ in range(config.vision_depth)
            ]
        )
        
        self.final_norm = RMSNorm(config.width)

    def forward(self, images: Tensor) -> Tensor:
        # Conv 输出 [B, D, H/P, W/P]；flatten+transpose 后成为 [B, N_patches, D]。
        x = self.patch_projection(images).flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        
        # 本示例所有 patch 都有效，所以 mask 全为 True。
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for layer in self.layers:
            # Vision 阶段不需要缓存 K/V，因此只保留第一个返回值 x。
            x, _, _ = layer(x, valid)
        return self.final_norm(x)


@dataclass
class PrefixKVCache:
    """逐层保存 Prefix K/V。

    ``keys[i]`` 和 ``values[i]`` 供第 i 层 Action Expert 使用，单个张量形状为：
    [batch, heads, prefix_tokens, head_dim]。
    """

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    valid: Tensor


def sinusoidal_time_embedding(time: Tensor, width: int) -> Tensor:
    """把标量 Flow 时间 t 编码成 width 维正余弦向量。

    time 的形状是 [B]，输出为 [B, width]。不同周期让网络既能感知 t 的粗粒度位置，
    也能区分相邻的细粒度时间。注意：这里的 t 不是机器人控制时间。
    """

    half = width // 2
    
    # 按对数间隔生成从 4e-3 到 4.0 的一组周期，覆盖不同时间尺度。
    periods = torch.logspace(
        math.log10(4e-3), math.log10(4.0), half, device=time.device, dtype=time.dtype
    )
    phase = 2.0 * math.pi * time[:, None] / periods[None, :]
    
    # sin/cos 各占一半通道，使时间编码同时携带相位的两个正交分量。
    embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    
    # width 为奇数时补 1 维；本示例默认 width=32，不会进入该分支。
    if embedding.shape[-1] < width:
        embedding = F.pad(embedding, (0, width - embedding.shape[-1]))
    return embedding


class TinyPi05(nn.Module):
    """完整教学模型：Tiny VLM Prefix + 条件 Flow Matching Action Expert。"""

    def __init__(self, config: TinyPi05Config):
        super().__init__()
        self.config = config

        # ---------- 不随 Flow 迭代变化的 Prefix 分支 ----------
        # 图像通过 Tiny SigLIP 得到视觉 token。
        self.vision_encoder = TinyVisionEncoder(config)
        
        # 整数文本 ID 通过查表变成连续向量，作用对应 Gemma token embedding。
        self.text_embedding = nn.Embedding(config.vocab_size, config.width) # 260,32
        
        # 图像/文本联合进入 Tiny Gemma，并在每层产生可缓存 K/V。
        self.prefix_layers = nn.ModuleList(
            [
                PrefixBlock(config.width, config.num_heads, config.mlp_dim)
                for _ in range(config.prefix_depth)
            ]
        )

        # ---------- 每个 Euler step 都会运行的 Action Expert 分支 ----------
        # 每个连续动作向量 [action_dim] 对应一个 action token [width]。
        self.action_in_projection = nn.Linear(config.action_dim, config.width) # 32, 32 
        
        # 正余弦时间编码再通过两层 MLP，作为所有 AdaRMS 的条件。
        self.time_mlp = nn.Sequential(
            nn.Linear(config.width, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
            nn.SiLU(),
        )
        
        self.expert_layers = nn.ModuleList(
            [
                ActionExpertBlock(config.width, config.num_heads, config.mlp_dim)
                for _ in range(config.expert_depth)
            ]
        )
        
        # 官方 π₀.₅ 的 Action Expert 最后一层同样使用时间条件 AdaRMS。
        self.expert_final_norm = AdaRMSNorm(config.width, config.width)
        
        # 把 Expert hidden state 投影回连续动作空间；输出就是速度场 v_t。
        self.action_out_projection = nn.Linear(config.width, config.action_dim)

    def encode_prefix(
        self,
        images: Tensor,
        image_valid: Tensor,
        token_ids: Tensor,
        token_valid: Tensor,
    ) -> PrefixKVCache:
        """联合编码图像和文本，并保留每一层 K/V。

        输入：
          images: [B, V=3, 3, 224, 224]
          image_valid: [B, V=3]，标记每个相机槽是否真的存在
          token_ids: [B, T]
          token_valid: [B, T]

        每个视角生成 256 个 token，所以最大 Prefix 长度是：

            3×256 个视觉 token + 200 个文本 token = 968

        输出：PrefixKVCache，其中每层 K/V 为 [B, heads, 968, head_dim]。
        推理时本函数只调用一次，结果供所有 Euler step 复用。
        """

        if images.ndim != 5:
            raise ValueError("images must have shape [batch, views, 3, height, width]")
        batch, views, channels, height, width = images.shape
        
        if views != self.config.num_image_views:
            raise ValueError(f"expected {self.config.num_image_views} image views, got {views}")
        expected_image_shape = (3, self.config.image_size, self.config.image_size)
        
        if (channels, height, width) != expected_image_shape:
            raise ValueError(
                f"each image must have shape {expected_image_shape}, "
                f"got {(channels, height, width)}"
            )
        if image_valid.shape != (batch, views):
            raise ValueError(f"image_valid must have shape {(batch, views)}, got {tuple(image_valid.shape)}")

        # 把 view 维并入 batch，共享同一个 Tiny SigLIP：
        # [B, 3, 3, 224, 224] -> [B*3, 3, 224, 224] -> [B*3, 256, 32]。
        flat_images = images.reshape(batch * views, channels, height, width)
        per_view_tokens = self.vision_encoder(flat_images) # 就是把图片变成 token，256 个 token 表示一张图，每个 token 的维度是 32[6, 256, 32]
        tokens_per_view = per_view_tokens.shape[1]
        
        if tokens_per_view != self.config.image_tokens_per_view:
            raise RuntimeError(
                f"expected {self.config.image_tokens_per_view} image tokens per view, "
                f"got {tokens_per_view}"
            )
        # 恢复 batch，并把 3 个相机槽沿序列维拼接：[B, 3*256, 32]。
        image_tokens = per_view_tokens.reshape(batch, views * tokens_per_view, self.config.width)

        # [B, 200] -> [B, 200, 32]；乘 sqrt(width) 是 Gemma embedding 的幅值约定。
        text_tokens = self.text_embedding(token_ids) * math.sqrt(self.config.width)
        
        # Prefix 序列布局：[全部 image tokens, 全部 text tokens]。 图像和文本的 token 拼接到一起
        x = torch.cat([image_tokens, text_tokens], dim=1)

        # 一个相机槽的 bool mask 要扩展到该视角的全部 256 个 patch token。
        expanded_image_valid = image_valid[:, :, None].expand(batch, views, tokens_per_view)
        expanded_image_valid = expanded_image_valid.reshape(batch, views * tokens_per_view)
        prefix_valid = torch.cat([expanded_image_valid, token_valid], dim=1)

        # 逐层运行 Prefix Transformer，并收集“该层实际用于 Attention 的 K/V”。
        keys: list[Tensor] = []
        values: list[Tensor] = []
        for layer in self.prefix_layers:
            x, k, v = layer(x, prefix_valid)
            keys.append(k)
            values.append(v)
        return PrefixKVCache(tuple(keys), tuple(values), prefix_valid)

    def predict_velocity(self, x_t: Tensor, time: Tensor, cache: PrefixKVCache) -> Tensor:
        r"""Action Expert 前向：预测条件速度场 :math:`v_\theta(x_t,t,c)`。

        输入：
          x_t: [B, H, action_dim]，Flow 时间 t 的带噪动作块
          time: [B]，当前 Flow 时间
          cache: 条件 c（图像+文本）对应的逐层 Prefix K/V

        输出：
          velocity: [B, H, action_dim]，表示 dx_t/dt，而不是机器人物理速度
        """

        # 连续动作不是离散词表 ID，直接用 Linear 映射成 H 个 action token。
        x = self.action_in_projection(x_t)
        
        # 标量 t -> 正余弦编码 -> 两层 MLP -> [B, width] AdaRMS 条件。
        time_condition = self.time_mlp(sinusoidal_time_embedding(time, self.config.width))

        # Expert 第 i 层读取 Prefix 第 i 层的 K/V；strict=True 防止层数静默错位。
        for layer, prefix_k, prefix_v in zip(
            self.expert_layers, cache.keys, cache.values, strict=True
        ):
            x = layer(x, time_condition, prefix_k, prefix_v, cache.valid)
        
        # 最后再做一次时间条件归一化，然后从 hidden width 投影回 action_dim。
        x, _ = self.expert_final_norm(x, time_condition)
        return self.action_out_projection(x)

    def flow_matching_loss(
        self,
        images: Tensor,
        image_valid: Tensor,
        token_ids: Tensor,
        token_valid: Tensor,
        actions: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""计算一批数据的 Conditional Flow Matching 训练损失。

        本代码沿用 OpenPI 源码的时间方向：t=1 为纯噪声，t=0 为真实动作。

        1. 真实动作块：A ~ p_data(A | image, text)
        2. 高斯噪声：epsilon ~ N(0, I)
        3. 线性概率路径：x_t = t*epsilon + (1-t)*A
        4. 路径对 t 的导数：u_t = dx_t/dt = epsilon - A
        5. 训练目标：MSE(v_theta(x_t, t, condition), u_t)

        训练时只随机抽一个 t 做一次网络前向，不会展开十步 Euler；十步循环只在推理时发生。
        """

        batch = actions.shape[0]
        
        # noise 与 actions 完全同形状：[B, action_horizon, action_dim]。
        if noise is None:
            noise = torch.randn_like(actions) #   actions = [2, 10, 32]
        if time is None:
            # Beta(1.5, 1) 的精确逆 CDF 采样，再缩放到 OpenPI 使用的 [0.001, 1]。
            # 它比均匀分布更偏向较大的 t，即训练时更常看到噪声较强的 x_t。
            time = torch.rand(batch, device=actions.device).pow(1.0 / 1.5)
            time = time * 0.999 + 0.001

        # [B] -> [B, 1, 1]，从而在 H 和 action_dim 两个维度广播。
        expanded_time = time[:, None, None]
        
        # t=1 时 x_t=noise；t=0 时 x_t=actions。
        x_t = expanded_time * noise + (1.0 - expanded_time) * actions
        
        # 因为采用直线路径，针对一对 (noise, action)，目标速度不随 t 改变。
        target_velocity = noise - actions

        # 训练时 Prefix 和 Expert 联合训练，所以这里不 detach cache。
        cache = self.encode_prefix(images, image_valid, token_ids, token_valid)
        predicted_velocity = self.predict_velocity(x_t, time, cache)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        # 附带返回中间量，方便调试和理解；detach 避免外部日志意外保留计算图。
        return loss, {
            "time": time.detach(),
            "x_t": x_t.detach(),
            "target_velocity": target_velocity.detach(),
            "predicted_velocity": predicted_velocity.detach(),
        }

    @torch.no_grad()
    def sample_actions(
        self,
        images: Tensor,
        image_valid: Tensor,
        token_ids: Tensor,
        token_valid: Tensor,
        *,
        noise: Tensor | None = None,
        num_steps: int = 10,
    ) -> tuple[Tensor, list[dict[str, float]]]:
        r"""用 Euler 法从 t=1 的噪声积分到 t=0 的动作块。

        待求解的常微分方程是：

            dx_t/dt = v_theta(x_t, t, condition)

        因为代码从 t=1 走向 t=0，所以 dt=-1/num_steps。每一步都要用更新后的 x_t
        和 t 重新预测局部速度，但图像/文本 Prefix K/V 始终复用。

        返回 ``(actions, trace)``：actions 为 [B, H, action_dim]，trace 记录每步数值范围。
        """

        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        batch = images.shape[0]
        if noise is None:
        
            # 随机初值决定生成分布中的一个样本；相同条件配不同噪声可得到不同合理动作。
            noise = torch.randn(
                batch,
                self.config.action_horizon,
                self.config.action_dim,
                device=images.device,
            )

        # 最贵的视觉/语言前向只执行一次；这就是 Prefix K/V Cache 的使用位置。
        cache = self.encode_prefix(images, image_valid, token_ids, token_valid)
        x_t = noise.clone()
        
        # 例如 num_steps=10，则 dt=-0.1，依次在 t=1.0, 0.9, ..., 0.1 计算速度。
        dt = -1.0 / num_steps
        trace: list[dict[str, float]] = []
        for step in range(num_steps):
            scalar_time = 1.0 - step / num_steps
            time = torch.full((batch,), scalar_time, device=images.device)
        
            # 每一步只重跑较小的 Action Expert；输入 x_t 和 t 均已变化。
            velocity = self.predict_velocity(x_t, time, cache)
        
            # Forward Euler：x(t+dt) ≈ x(t) + dt * v(x(t), t)。
            x_t = x_t + dt * velocity
        
            # RMS 只用于观察数值是否发散，不参与模型输出和下一步计算。
            trace.append(
                {
                    "step": float(step + 1),
                    "time": scalar_time,
                    "x_rms": float(x_t.float().square().mean().sqrt()),
                    "v_rms": float(velocity.float().square().mean().sqrt()),
                }
            )
        return x_t, trace


def make_toy_batch(
    config: TinyPi05Config,
    device: torch.device,
) -> tuple[list[str], Tensor, Tensor, Tensor]:
    """构造一个确定性的 Toy 模仿学习 batch，不依赖 LIBERO 数据集。

    任务只有“向左”和“向右”两类。文本、图像亮区和动作方向彼此一致，使我们可以验证：

    - 文本能否被编码；
    - 图像能否变成 patch token；
    - Flow Matching loss 能否反向传播；
    - 采样器能否输出官方内部形状 [B, 10, 32] 的动作块。

    这不是机器人动力学数据，因此不能用生成动作判断官方模型精度。
    """

    prompts = [
        "move the gripper left",
        "move the gripper right",
    ]
    batch = len(prompts)
    
    # OpenPI 固定接收三个相机槽：[base, left_wrist, right_wrist]。
    # PyTorch 教学版采用 channel-first，形状为 [B, 3 views, 3 channels, 224, 224]。
    images = torch.zeros(
        batch,
        config.num_image_views,
        3,
        config.image_size,
        config.image_size,
        device=device,
    )
    
    # 主相机：左移样本左半边亮，右移样本右半边亮。
    images[0, 0, :, :, : config.image_size // 2] = 1.0
    images[1, 0, :, :, config.image_size // 2 :] = 1.0
    
    # 左腕相机提供一个较弱的共同信号；右腕槽保持全零，模拟 LIBERO 缺少第二腕部相机。
    images[:, 1, :, config.image_size // 4 : config.image_size // 2, :] = 0.5
    image_valid = torch.tensor(
        [[True, True, False], [True, True, False]],
        dtype=torch.bool,
        device=device,
    )

    # OpenPI 内部 actions: [B, 10, 32]。LIBERO 的真实 7 维放在最前面，其余维度补零。
    actions = torch.zeros(batch, config.action_horizon, config.action_dim, device=device) # [2, 10, 32]
    
    # 从 0.1 到 1.0 的序列模拟动作幅度逐步增加。
    trajectory = torch.linspace(0.1, 1.0, config.action_horizon, device=device) # [10] 个循环的step scale
    
    # 第 0 维：偶数样本向左（负），奇数样本向右（正）。
    actions[0::2, :, 0] = -trajectory
    actions[1::2, :, 0] = trajectory
    
    # 第 1 维放一个较小的共同运动；LIBERO 有效区间的最后一维（索引 6）作为夹爪命令。
    actions[:, :, 1] = 0.25 * trajectory
    actions[:, :, config.libero_action_dim - 1] = 1.0
    return prompts, images, image_valid, actions


def run_demo(args: argparse.Namespace) -> None:
    """端到端演示：造数据 -> Tokenize -> 训练 -> 缓存 Prefix -> Euler 采样。"""

    # 固定随机种子，确保初始化、训练噪声和输出尽量可复现。
    torch.manual_seed(args.seed)
    
    # Tiny 矩阵用过多 CPU 线程反而会增加调度开销。
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device(args.device)

    # 1) 创建配置、Tokenizer 和模型。
    config = TinyPi05Config()
    tokenizer = TinyTextTokenizer(config.max_text_tokens)
    model = TinyPi05(config).to(device)

    # 2) 构造 Toy 观测/动作，并把字符串处理成定长 token ID 与 mask。
    prompts, images, image_valid, target_actions = make_toy_batch(config, device)  # [2], [2, 3, 3, 224, 224], [2, 3], [2, 10, 32]
    token_ids, token_valid = tokenizer.batch_encode(prompts, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model parameters: {parameter_count:,}")
    print(
        f"patch grid:       {config.patch_grid_size}x{config.patch_grid_size} "
        f"= {config.image_tokens_per_view} tokens/view"
    )
    print(f"max prefix:       {config.max_prefix_tokens} tokens")
    print(f"images:           {tuple(images.shape)}")
    print(f"image valid:      {tuple(image_valid.shape)}")
    print(f"text token IDs:   {tuple(token_ids.shape)}")
    print(f"target actions:   {tuple(target_actions.shape)}")

    # 3) Flow Matching 训练。每次迭代都会重新采样 noise 和 t。
    model.train()
    first_loss = None
    for step in range(args.train_steps):
        # 清除上一步梯度；set_to_none=True 比填零更省内存，也能暴露未收到梯度的参数。
        optimizer.zero_grad(set_to_none=True)
        loss, values = model.flow_matching_loss(
            images, image_valid, token_ids, token_valid, target_actions
        )
        
        # 从 velocity MSE 沿 Action Expert、Prefix Gemma、Vision Encoder 反向传播。
        loss.backward()
        optimizer.step()
        detached_loss = float(loss.detach())
        first_loss = detached_loss if first_loss is None else first_loss
        if step == 0 or (step + 1) % max(1, args.train_steps // 5) == 0:
            print(
                f"train {step + 1:4d}/{args.train_steps}: "
                f"loss={detached_loss:.6f}, mean(t)={float(values['time'].mean()):.3f}"
            )

    # 4) 切换推理模式，单独打印 Prefix Cache 形状帮助核对数据流。
    model.eval()
    cache = model.encode_prefix(images, image_valid, token_ids, token_valid)
    print(f"prefix valid mask:{tuple(cache.valid.shape)}")
    print(f"layer-0 cached K: {tuple(cache.keys[0].shape)}")

    # 5) 使用固定初始噪声执行 Euler 采样，便于比较修改前后的结果。
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial_noise = torch.randn(
        len(prompts), config.action_horizon, config.action_dim, generator=generator, device=device
    )
    
    sampled_actions, trace = model.sample_actions(
        images,
        image_valid,
        token_ids,
        token_valid,
        noise=initial_noise,
        num_steps=args.flow_steps,
    )
    
    # 数值爆炸时尽早失败，而不是继续把 NaN/Inf 当作机器人动作使用。
    if not torch.isfinite(sampled_actions).all():
        raise RuntimeError("non-finite action produced")

    print(f"sampled actions:  {tuple(sampled_actions.shape)}")
    print("Euler trace (same Prefix K/V reused at every row):")
    
    for item in trace:
        print(
            f"  step={int(item['step']):2d}  t={item['time']:.2f}  "
            f"x_rms={item['x_rms']:.4f}  v_rms={item['v_rms']:.4f}"
        )
    
    print("first generated action chunk, first 3 rows, valid LIBERO 7 dims only:")
    print(sampled_actions[0, :3, : config.libero_action_dim].cpu())
    if first_loss is not None:
        print(f"first training loss: {first_loss:.6f}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并在运行模型前完成基本合法性检查。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=2, help="Toy Flow Matching 训练步数")
    parser.add_argument("--flow-steps", type=int, default=10, help="推理时 Euler 积分步数")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="AdamW 学习率")
    parser.add_argument("--device", default="cpu", help="例如 cpu、cuda、cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2, help="CPU 模式使用的 Torch 线程数")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    args = parser.parse_args()
    
    if args.train_steps < 0:
        parser.error("--train-steps must be >= 0")
    
    if args.flow_steps <= 0:
        parser.error("--flow-steps must be > 0")
    
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be > 0")
    
    return args


if __name__ == "__main__":
    
    # 只有直接执行本文件时才运行 demo；被其他模块 import 时只提供类和函数定义。
    run_demo(parse_args())
