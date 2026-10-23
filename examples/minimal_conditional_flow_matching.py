#!/usr/bin/env python3
r"""用二维 Toy 动作分布解释 Conditional Flow Matching（CFM）。

这个示例刻意不使用图像、Gemma 或机器人数据，只保留 π₀/π₀.₅ Action Expert
最核心的训练和采样逻辑：

    条件 c + 带噪动作 x_t + Flow 时间 t -> 条件速度场 v_theta(x_t, t, c)

Toy 数据有两个语言条件：

    c=0（left）  -> 合理动作位于 (-2, +1) 或 (-2, -1)
    c=1（right） -> 合理动作位于 (+2, +1) 或 (+2, -1)

每个条件故意保留“上/下”两个合理动作模式。只做普通 MSE 动作回归时，最优预测会是
两个模式的平均值，例如 left -> (-2, 0)，但训练数据中并没有这个动作。Flow Matching
则把初始高斯噪声当作随机性来源，可以在相同条件下生成上、下两种结果。

本文件采用 OpenPI 源码的时间方向：

    t=0：真实动作 A
    t=1：高斯噪声 epsilon

训练：

    x_t = (1-t) * A + t * epsilon
    u_t = d(x_t)/dt = epsilon - A
    loss = MSE(v_theta(x_t, t, c), u_t)

推理：从 t=1 的噪声出发，用负步长 Euler 积分回到 t=0：

    x <- x + dt * v_theta(x, t, c),  dt < 0

运行：

    python examples/minimal_conditional_flow_matching.py

想对照最基础的均匀时间采样：

    python examples/minimal_conditional_flow_matching.py --time-sampling uniform
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DemoConfig:
    """Toy 数据、速度场网络和训练过程的集中配置。"""

    action_dim: int = 2
    num_conditions: int = 2
    hidden_dim: int = 64
    target_x: float = 2.0
    target_y: float = 1.0
    data_std: float = 0.10


def sample_conditional_actions(
    batch_size: int,
    config: DemoConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """从人为定义的条件动作分布 p(A|c) 中采样。

    返回：
      actions:    [B, 2]，二维连续动作
      conditions: [B]，0 表示 left，1 表示 right

    条件只决定左右方向，没有决定上下分支。因此同一个条件下存在两种合理动作，
    这正是生成模型比“直接预测条件均值”更有价值的场景。
    """

    conditions = torch.randint(0, config.num_conditions, (batch_size,), device=device)
    branches = torch.randint(0, 2, (batch_size,), device=device)

    # c=0 -> x=-2，c=1 -> x=+2。
    center_x = torch.where(
        conditions == 0,
        -torch.full((batch_size,), config.target_x, device=device),
        torch.full((batch_size,), config.target_x, device=device),
    )
    # branch=0 -> y=-1，branch=1 -> y=+1。
    center_y = torch.where(
        branches == 0,
        -torch.full((batch_size,), config.target_y, device=device),
        torch.full((batch_size,), config.target_y, device=device),
    )

    centers = torch.stack([center_x, center_y], dim=-1)
    # 给每个模式添加少量连续扰动，使目标不是四个完全离散的点。
    actions = centers + config.data_std * torch.randn_like(centers)
    return actions, conditions


def sample_flow_time(
    batch_size: int,
    device: torch.device,
    mode: str,
) -> Tensor:
    """为每个动作样本采一个 Flow 时间，返回形状 [B]。

    ``uniform`` 是最基础的 Flow Matching 采样。``openpi`` 复现 OpenPI 的
    Beta(1.5, 1) 与 ``*0.999+0.001`` 设置，更偏向 t 较大的噪声区域。
    """

    uniform = torch.rand(batch_size, device=device)
    if mode == "uniform":
        return uniform
    if mode == "openpi":
        # 若 U~Uniform(0,1)，则 U^(1/alpha)~Beta(alpha,1)。
        return uniform.pow(1.0 / 1.5) * 0.999 + 0.001
    raise ValueError(f"unknown time sampling mode: {mode}")


class ConditionalVelocityField(nn.Module):
    r"""一个很小的 MLP，表示条件速度场 v_theta(x_t, t, c)。

    对二维平面上的任意位置 x_t、Flow 时间 t 和离散条件 c，模型输出一个二维箭头：

        [x方向速度, y方向速度]

    这两个数不是机器人真实世界中的物理速度，而是动作向量关于人工 Flow 时间的导数。
    """

    def __init__(self, config: DemoConfig):
        super().__init__()
        input_dim = config.action_dim + 1 + config.num_conditions
        self.num_conditions = config.num_conditions
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(self, x_t: Tensor, time: Tensor, conditions: Tensor) -> Tensor:
        """预测速度；输入 shape 分别为 [B,2]、[B]、[B]，输出为 [B,2]。"""

        # one_hot(0)=[1,0]，one_hot(1)=[0,1]，让 MLP 明确区分两个任务条件。
        condition_features = F.one_hot(conditions, self.num_conditions).to(x_t.dtype)
        inputs = torch.cat(
            [x_t, time[:, None].to(x_t.dtype), condition_features],
            dim=-1,
        )
        return self.network(inputs)


def conditional_flow_matching_loss(
    model: ConditionalVelocityField,
    actions: Tensor,
    conditions: Tensor,
    *,
    time_sampling: str,
) -> tuple[Tensor, dict[str, Tensor]]:
    """构造 CFM 训练样本，并让模型速度匹配直线路径的目标速度。"""

    batch_size = actions.shape[0]

    # 路径终点之一：与动作完全同形状的标准高斯噪声 epsilon~N(0,I)。
    noise = torch.randn_like(actions)
    # 每个二维动作整体使用一个 Flow 时间，而不是两个坐标各用一个时间。
    time = sample_flow_time(batch_size, actions.device, time_sampling)

    # [B] -> [B,1]，使 time 同时作用到二维动作的 x/y 坐标。
    expanded_time = time[:, None]
    # 在真实动作 A 和噪声 epsilon 之间随机取一个中间点。
    x_t = (1.0 - expanded_time) * actions + expanded_time * noise
    # 直线路径 x_t=A+t(epsilon-A) 对 t 的导数是 epsilon-A。
    target_velocity = noise - actions

    predicted_velocity = model(x_t, time, conditions)
    loss = F.mse_loss(predicted_velocity, target_velocity)
    return loss, {
        "x_t": x_t.detach(),
        "time": time.detach(),
        "target_velocity": target_velocity.detach(),
        "predicted_velocity": predicted_velocity.detach(),
    }


def train_velocity_field(
    args: argparse.Namespace,
    config: DemoConfig,
    device: torch.device,
) -> ConditionalVelocityField:
    """重复生成 CFM 监督样本，训练出条件速度场。"""

    model = ConditionalVelocityField(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for step in range(1, args.train_steps + 1):
        actions, conditions = sample_conditional_actions(args.batch_size, config, device)
        loss, values = conditional_flow_matching_loss(
            model,
            actions,
            conditions,
            time_sampling=args.time_sampling,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.print_every == 0 or step == args.train_steps:
            print(
                f"train {step:5d}/{args.train_steps}: "
                f"loss={float(loss.detach()):.6f}, "
                f"mean(t)={float(values['time'].mean()):.3f}"
            )
    return model


@torch.no_grad()
def sample_with_euler(
    model: ConditionalVelocityField,
    conditions: Tensor,
    initial_noise: Tensor,
    num_steps: int,
) -> Tensor:
    """从 t=1 的高斯噪声反向积分到 t=0，生成条件动作。"""

    x_t = initial_noise.clone()
    dt = -1.0 / num_steps
    batch_size = x_t.shape[0]

    for step in range(num_steps):
        scalar_time = 1.0 - step / num_steps
        time = torch.full((batch_size,), scalar_time, device=x_t.device)
        velocity = model(x_t, time, conditions)
        x_t = x_t + dt * velocity
    return x_t


def nearest_mode_distance(samples: Tensor, condition: int, config: DemoConfig) -> Tensor:
    """计算每个生成动作到对应条件下最近真实模式中心的欧氏距离。"""

    x = -config.target_x if condition == 0 else config.target_x
    modes = torch.tensor(
        [[x, -config.target_y], [x, config.target_y]],
        dtype=samples.dtype,
        device=samples.device,
    )
    distances = torch.linalg.vector_norm(samples[:, None, :] - modes[None, :, :], dim=-1)
    return distances.min(dim=1).values


def print_distribution_summary(
    name: str,
    samples: Tensor,
    condition: int,
    config: DemoConfig,
) -> None:
    """打印生成分布是否到达正确左右区域、是否覆盖上下两个模式。"""

    expected_left = condition == 0
    side_correct = samples[:, 0] < 0 if expected_left else samples[:, 0] > 0
    upper_ratio = (samples[:, 1] > 0).float().mean()
    middle_ratio = (samples[:, 1].abs() < 0.5).float().mean()
    distances = nearest_mode_distance(samples, condition, config)

    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    print(f"\n{name}:")
    print(f"  mean(x,y)              = ({float(mean[0]):+.3f}, {float(mean[1]):+.3f})")
    print(f"  std(x,y)               = ({float(std[0]):.3f}, {float(std[1]):.3f})")
    print(f"  requested-side accuracy= {float(side_correct.float().mean()):.3f}")
    print(f"  upper-mode ratio       = {float(upper_ratio):.3f}  (理想值约 0.5)")
    print(f"  middle-collapse ratio  = {float(middle_ratio):.3f}  (越低越好)")
    print(f"  nearest-mode distance  = {float(distances.mean()):.3f}  (越低越好)")


@torch.no_grad()
def evaluate_model(
    model: ConditionalVelocityField,
    args: argparse.Namespace,
    config: DemoConfig,
    device: torch.device,
) -> None:
    """验证条件是否改变生成结果，以及随机噪声是否保留多模态。"""

    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial_noise = torch.randn(
        args.eval_samples,
        config.action_dim,
        generator=generator,
        device=device,
    )
    left_conditions = torch.zeros(args.eval_samples, dtype=torch.long, device=device)
    right_conditions = torch.ones(args.eval_samples, dtype=torch.long, device=device)

    # 故意对左右条件复用同一批初始噪声，从而隔离并观察条件 c 的影响。
    left_actions = sample_with_euler(
        model, left_conditions, initial_noise, args.flow_steps
    )
    right_actions = sample_with_euler(
        model, right_conditions, initial_noise, args.flow_steps
    )

    print_distribution_summary("condition=left", left_actions, 0, config)
    print_distribution_summary("condition=right", right_actions, 1, config)

    print("\n同一初始噪声在不同条件下的前 5 个输出：")
    for index in range(min(5, args.eval_samples)):
        noise = initial_noise[index]
        left = left_actions[index]
        right = right_actions[index]
        print(
            f"  noise=({float(noise[0]):+.2f},{float(noise[1]):+.2f})  "
            f"left=({float(left[0]):+.2f},{float(left[1]):+.2f})  "
            f"right=({float(right[0]):+.2f},{float(right[1]):+.2f})"
        )

    # 若直接用 MSE 从 condition 回归动作且没有额外随机变量，最优解是条件均值。
    direct_left = torch.tensor([-config.target_x, 0.0], device=device)
    direct_right = torch.tensor([config.target_x, 0.0], device=device)
    direct_distance = config.target_y
    print("\n直接 MSE 动作回归的理论条件均值：")
    print(f"  left  -> ({float(direct_left[0]):+.1f}, {float(direct_left[1]):+.1f})")
    print(f"  right -> ({float(direct_right[0]):+.1f}, {float(direct_right[1]):+.1f})")
    print(
        f"  它们到最近真实模式的距离都是 {direct_distance:.1f}，"
        "说明确定性均值回归落在两个模式之间。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--time-sampling", choices=("openpi", "uniform"), default="openpi")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    for name in ("train_steps", "batch_size", "flow_steps", "eval_samples", "print_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    config = DemoConfig()

    print("目标条件分布：")
    print("  left : (-2,-1) 或 (-2,+1)")
    print("  right: (+2,-1) 或 (+2,+1)")
    print(f"time sampling: {args.time_sampling}")

    model = train_velocity_field(args, config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"velocity-field parameters: {parameter_count:,}")
    evaluate_model(model, args, config, device)


if __name__ == "__main__":
    main()
