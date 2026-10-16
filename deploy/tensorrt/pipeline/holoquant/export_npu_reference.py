#!/usr/bin/env python3
"""给 HoloQuant NPU 仿真导出 π₀.₅ 的配置、FP8 scale、样本和参考答案。

本脚本属于 OpenPI 仓库：只 import OpenPI / TensorRT / MTQ（Python 3.11 + torch 2.7），
不 import holoq。产物全部是 json / npz，HoloQuant 的 02 / 03 只读文件。
权重不在这里导出，HoloQuant 的 02 直接读 checkpoint 的 model.safetensors。

产物（默认 ``<workspace>/Outputs/HoloQuant/pi05_libero/openpi_export/``）::

    config.json             模型超参：层数、头数、eps、RoPE base、mask 值……
    constants.npz           RoPE 的 inv_freq、时间编码的频率表（编译期常量）
    fp8_amax.json           每个 FP8 Linear 的 input / weight amax
                              Prefix / Denoise：MTQ FP8_DEFAULT_CFG，与 TRT mtqfp8 同一配方
                              Vision：forward hook 统计 max|x|（MTQ 配方里 vision 不量化）
    samples/sample_XXX.npz  一条样本：模型输入、bf16 PyTorch 的中间结果和动作，
                            以及 MTQ 假量化动作（L1）、TensorRT 动作（--trt）
    summary.json            L1 / TRT 相对 bf16 的动作 RMSE

在 OpenPI 仓库根执行::

    ./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh
    ./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --trt
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _looks_like_openpi(root: Path) -> bool:
    """OpenPI 仓库根：有 src/openpi，且有本脚本要读的 deploy/tensorrt。"""
    return (root / 'src' / 'openpi').is_dir() and (root / 'deploy' / 'tensorrt').is_dir()


def find_openpi_root(explicit: Path | None = None) -> Path:
    """定位本仓库根。脚本已经在 OpenPI 里，默认沿 __file__ 往上走即可。"""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if _looks_like_openpi(root):
            return root
        raise SystemExit(f'--openpi-root={root} 不像 OpenPI 仓库（需要 src/openpi 和 deploy/tensorrt）。')

    env = os.environ.get('OPENPI_ROOT')
    if env:
        root = Path(env).expanduser().resolve()
        if _looks_like_openpi(root):
            return root
        raise SystemExit(f'OPENPI_ROOT={root} 不像 OpenPI 仓库。')

    for parent in Path(__file__).resolve().parents:
        if _looks_like_openpi(parent):
            return parent

    raise SystemExit(
        '找不到 OpenPI 仓库根。本脚本应放在 deploy/tensorrt/pipeline/holoquant/ 下。\n'
        '请在仓库根执行：\n'
        '  ./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh'
    )


def default_out_dir(openpi_root: Path) -> Path:
    """两边仓库约定的交换目录，不 import HoloQuant。

    顺序：``HOLOQUANT_EXPORT_DIR`` → 沿仓库往上找 ``Outputs/HoloQuant``
    → 本机默认路径。
    """
    env = os.environ.get('HOLOQUANT_EXPORT_DIR')
    if env:
        return Path(env).expanduser().resolve()
    for ancestor in [openpi_root, *openpi_root.parents]:
        holo_root = ancestor / 'Outputs' / 'HoloQuant'
        if holo_root.is_dir():
            return holo_root / 'pi05_libero' / 'openpi_export'
    return Path('/share_data/bruce/workspace/Outputs/HoloQuant/pi05_libero/openpi_export')


def _load_file_module(path: Path, name: str):
    """按文件路径加载模块。OpenPI 的 02_dump_calib.py 等文件名以数字开头，不能直接 import。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def model_config(model, lang_len: int) -> dict:
    pg = model.paligemma_with_expert
    vis = pg.paligemma.config.vision_config
    txt = pg.paligemma.config.text_config
    exp = pg.gemma_expert.config

    def gemma(cfg, layer) -> dict:
        return dict(
            width=cfg.hidden_size, depth=cfg.num_hidden_layers, mlp_dim=cfg.intermediate_size,
            num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, rms_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta,
            hidden_act=cfg.hidden_activation, attn_scaling=float(layer.self_attn.scaling),
        )

    vis_layer = pg.paligemma.model.vision_tower.vision_model.encoder.layers[0]
    num_views = 3
    vision_tokens = (vis.image_size // vis.patch_size) ** 2
    return dict(
        vision=dict(
            width=vis.hidden_size, depth=vis.num_hidden_layers, mlp_dim=vis.intermediate_size,
            num_heads=vis.num_attention_heads, head_dim=vis.hidden_size // vis.num_attention_heads,
            patch=vis.patch_size, image=vis.image_size, ln_eps=vis.layer_norm_eps,
            hidden_act=vis.hidden_act, projection_dim=vis.projection_dim,
            attn_scaling=float(vis_layer.self_attn.scale),
        ),
        prefix=gemma(txt, pg.paligemma.model.language_model.layers[0]),
        expert=gemma(exp, pg.gemma_expert.model.layers[0]),
        action_dim=model.config.action_dim,
        action_horizon=model.config.action_horizon,
        num_steps=10,
        num_views=num_views,
        vision_tokens=vision_tokens,
        lang_len=lang_len,
        prefix_len=num_views * vision_tokens + lang_len,
        lang_emb_scale=math.sqrt(txt.hidden_size),
        mask_neg=-2.3819763e38,
        time_embed=dict(dim=model.action_in_proj.out_features, min_period=4e-3, max_period=4.0),
    )


def time_embed_freq(dim: int, min_period: float, max_period: float) -> np.ndarray:
    """create_sinusoidal_pos_embedding 的频率表 2π/period，float64，编译期常量。"""
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction
    return 1.0 / period * 2 * math.pi


def make_obs(policy, device: str, frame: np.ndarray | None = None):
    """和 05_compare_actions / run_step2 相同的随机 LIBERO 观测；frame 不为空时换成真实视频帧。"""
    import jax
    from openpi.models import model as openpi_model
    from openpi.policies.libero_policy import make_libero_example

    example = make_libero_example()
    if frame is not None:
        example['observation/image'] = frame
        example['observation/wrist_image'] = frame
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...],
        policy._input_transform(example),
    )
    return openpi_model.Observation.from_dict(inputs)


@torch.no_grad()
def capture_reference(model, observation, noise: torch.Tensor, num_steps: int) -> dict:
    """按 sample_actions 的顺序跑一遍 bf16 模型，把每张子图的输入输出都留下来。"""
    from deploy.tensorrt.common import NUM_LAYERS, cache_to_kv_tensor
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    device = noise.device
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
    prefix_mask_4d = model._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad, prefix_att))
    pg = model.paligemma_with_expert
    pg.paligemma.language_model.config._attn_implementation = 'eager'  # noqa: SLF001
    _, past = pg.forward(
        attention_mask=prefix_mask_4d, position_ids=prefix_pos, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True,
    )
    past_kv = cache_to_kv_tensor(past, NUM_LAYERS)  # [L, 2, B, H, S, D]

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    t = torch.tensor(1.0, dtype=torch.float32, device=device)
    steps = {k: [] for k in ('x_t', 'time', 'suffix_embs', 'adarms_cond', 'v_t')}
    while t >= -dt / 2:
        expanded = t.expand(x_t.shape[0])
        suffix_embs, _, _, adarms_cond = model.embed_suffix(state, x_t, expanded)
        v_t = model.denoise_step(state, prefix_pad, past, x_t, expanded)
        steps['x_t'].append(_f32(x_t[0]))
        steps['time'].append(np.float32(t.item()))
        steps['suffix_embs'].append(_f32(suffix_embs[0]))
        steps['adarms_cond'].append(_f32(adarms_cond[0]))
        steps['v_t'].append(_f32(v_t[0]))
        x_t = x_t + dt * v_t
        t = t + dt

    return dict(
        images=np.stack([_f32(img[0]) for img in images]),
        img_masks=np.array([bool(m[0]) for m in img_masks]),
        lang_tokens=lang_tokens[0].cpu().numpy().astype(np.int64),
        lang_masks=lang_masks[0].cpu().numpy().astype(bool),
        noise=_f32(noise[0]),
        ref_prefix_embs=_f32(prefix_embs[0]),
        ref_prefix_pad=prefix_pad[0].cpu().numpy().astype(bool),
        ref_past_kv=_f32(past_kv[:, :, 0]),            # [L, 2, H, S, D]
        ref_x_t=np.stack(steps['x_t']),                 # [T, horizon, action_dim]，第 k 步的输入
        ref_time=np.stack(steps['time']),               # [T]
        ref_suffix_embs=np.stack(steps['suffix_embs']),
        ref_adarms_cond=np.stack(steps['adarms_cond']),
        ref_v_t=np.stack(steps['v_t']),
        ref_actions=_f32(x_t[0]),
    )


@torch.no_grad()
def vision_amax(model, image_batches: list[torch.Tensor]) -> dict:
    """Vision 不在 MTQ 配方里。用 forward hook 统计每个 Linear 输入的 max|x|，权重取 max|w|。"""
    pg = model.paligemma_with_expert.paligemma.model
    names = {mod: name for name, mod in model.named_modules()}
    stats: dict[str, float] = {}
    hooks = []
    for mod in list(pg.vision_tower.modules()) + [pg.multi_modal_projector.linear]:
        if not isinstance(mod, torch.nn.Linear):
            continue
        name = names[mod]

        def hook(_m, inp, name=name):
            amax = float(inp[0].detach().abs().amax().float())
            stats[name] = max(stats.get(name, 0.0), amax)

        hooks.append(mod.register_forward_pre_hook(hook))
    try:
        for batch in image_batches:
            model.paligemma_with_expert.embed_image(batch)
    finally:
        for h in hooks:
            h.remove()
    mods = dict(model.named_modules())
    return {
        name: dict(input_amax=amax, weight_amax=float(mods[name].weight.detach().abs().amax().float()))
        for name, amax in stats.items()
    }


def mtq_amax(model, calib: list, device: str, openpi_root: Path) -> dict:
    """与 run_step2 / run_step3 相同的 MTQ 配方，同时量化 Prefix 和 Denoise 的 Linear。

    MTQ 校准时量化器只统计、不量化，所以两段一起校准和 TRT 分两次校准得到的 amax 一样。
    """
    import modelopt.torch.quantization as mtq

    s2 = _load_file_module(
        openpi_root / 'deploy/tensorrt/pipeline/fp8/experiments/run_step2_mtq_fakequant.py', 's2_mtq')
    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for pattern in s2._ALWAYS_DISABLE:
        s2._append_disable(quant_cfg, pattern)

    def forward_loop(mdl) -> None:
        mdl.eval()
        with torch.no_grad():
            for i, (obs, noise) in enumerate(calib):
                mdl.sample_actions(device, obs, noise=noise.clone())
                if (i + 1) % 4 == 0 or i + 1 == len(calib):
                    print(f'  calib {i + 1}/{len(calib)}')

    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    out = {}
    for name, mod in model.named_modules():
        iq = getattr(mod, 'input_quantizer', None)
        wq = getattr(mod, 'weight_quantizer', None)
        if iq is None or wq is None or not iq.is_enabled:
            continue
        out[name] = dict(input_amax=float(iq.amax), weight_amax=float(wq.amax))
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def share_outputs(root: Path) -> None:
    """产物给 users 组和同事读写。临时目录 / umask 常把新目录收成 700。"""
    if not root.exists():
        return
    for dirpath, _, filenames in os.walk(root):
        try:
            os.chmod(dirpath, 0o2777)
        except OSError:
            pass
        for name in filenames:
            try:
                os.chmod(os.path.join(dirpath, name), 0o666)
            except OSError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint-dir', default=None,
                    help='默认用 deploy.tensorrt.common.DEFAULT_CHECKPOINT')
    ap.add_argument('--config-name', default=None)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num-samples', type=int, default=4, help='评测样本数（RESULTS.md 的尺子是 4 条）')
    ap.add_argument('--num-calib', type=int, default=16, help='MTQ 校准样本数（与 run_step2/3 相同）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--obs', choices=('random', 'video'), default='random',
                    help='random：随机 LIBERO 观测，和 05_compare_actions 同一把尺子；video：用 LIBERO 视频第一帧')
    ap.add_argument('--trt', action='store_true', help='同一批样本再跑一遍 TensorRT mtqfp8，作为对照组')
    ap.add_argument('--out-dir', type=Path, default=None,
                    help='默认写到 workspace/Outputs/HoloQuant/pi05_libero/openpi_export')
    ap.add_argument('--openpi-root', type=Path, default=None,
                    help='一般不用。脚本已在本仓库里，省略时沿文件往上找仓库根')
    args = ap.parse_args()

    openpi_root = find_openpi_root(args.openpi_root)
    print(f'[export] OpenPI 仓库: {openpi_root}', flush=True)
    if str(openpi_root) not in sys.path:
        sys.path.insert(0, str(openpi_root))
    try:
        from deploy.tensorrt.common import (
            ARTIFACTS,
            CALIB_DIR,
            DEFAULT_CHECKPOINT,
            DEFAULT_CONFIG,
            inductor_cache_env,
        )
        inductor_cache_env()
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as train_config
    except ImportError as exc:
        venv_py = openpi_root / '.venv' / 'bin' / 'python'
        raise SystemExit(
            f'已找到仓库 {openpi_root}，但当前解释器 import 失败：{exc}\n'
            f'必须用 OpenPI 环境（本机是 {venv_py}），不要用 conda torch212 / 系统 python。\n'
            f'正确示例：\n'
            f'  cd {openpi_root}\n'
            f'  {venv_py} {Path(__file__).resolve()} --trt'
        ) from exc

    checkpoint_dir = args.checkpoint_dir or str(DEFAULT_CHECKPOINT)
    config_name = args.config_name or DEFAULT_CONFIG
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out_dir(openpi_root)
    (out_dir / 'samples').mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = train_config.get_config(config_name)
    object.__setattr__(cfg.model, 'pytorch_compile_mode', None)
    policy = _policy_config.create_trained_policy(cfg, checkpoint_dir, pytorch_device=args.device)
    model = policy._model
    model.eval()
    print(f'[export] 模型加载完成 {time.time() - t0:.0f}s，float32 matmul precision={torch.get_float32_matmul_precision()}')

    frames = None
    if args.obs == 'video':
        dump = _load_file_module(openpi_root / 'deploy/tensorrt/pipeline/fp8/02_dump_calib.py', 'dump_calib')
        video_root = openpi_root / 'data/libero/videos_libero_spatial_pytorch'
        frames = lambda i: dump._maybe_video_frame(video_root, i)  # noqa: E731

    # ---- 1) 评测样本 + bf16 参考 ----
    np.random.seed(args.seed)
    noise_rng = np.random.default_rng(args.seed)
    samples, observations = [], []
    for i in range(args.num_samples):
        obs = make_obs(policy, args.device, frames(i) if frames else None)
        noise = torch.from_numpy(
            noise_rng.standard_normal((1, cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
        ).to(args.device)
        cap = capture_reference(model, obs, noise, num_steps=10)
        if i == 0:
            with torch.no_grad():
                direct = model.sample_actions(args.device, obs, noise=noise.clone())
            gap = float(np.abs(_f32(direct[0]) - cap['ref_actions']).max())
            print(f'[export] 逐步捕获 vs sample_actions 最大差 {gap:.2e}（应为 0）')
        samples.append(cap)
        observations.append((obs, noise))
        print(f'[export] 参考样本 {i + 1}/{args.num_samples}  |actions|max={np.abs(cap["ref_actions"]).max():.3f}')

    lang_len = int(samples[0]['lang_tokens'].shape[0])
    config = model_config(model, lang_len)
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')

    pg = model.paligemma_with_expert
    np.savez(
        out_dir / 'constants.npz',
        prefix_inv_freq=_f32(pg.paligemma.model.language_model.rotary_emb.inv_freq),
        expert_inv_freq=_f32(pg.gemma_expert.model.rotary_emb.inv_freq),
        time_freq=time_embed_freq(**{k: config['time_embed'][k] for k in ('dim', 'min_period', 'max_period')}),
    )

    # ---- 2) 校准数据（与 run_step3 相同：16 条随机观测，噪声种子 123）----
    np.random.seed(123)
    calib_rng = np.random.default_rng(123)
    calib = []
    for i in range(args.num_calib):
        obs = make_obs(policy, args.device, frames(1000 + i) if frames else None)
        noise = torch.from_numpy(calib_rng.standard_normal((1, 10, 32)).astype(np.float32)).to(args.device)
        calib.append((obs, noise))

    # ---- 3) Vision amax：校准观测的图像 + 02_dump_calib 存下的真实视频帧 ----
    batches = [img for obs, _ in calib for img in model._preprocess_observation(obs, train=False)[0]]
    real = CALIB_DIR / 'pixel_values.npy'
    if real.is_file():
        pix = np.load(real)
        for j in range(0, len(pix), 8):
            batches.append(torch.from_numpy(pix[j:j + 8].astype(np.float32)).to(args.device))
    amax = {'vision': vision_amax(model, batches)}
    print(f'[export] vision amax：{len(amax["vision"])} 个 Linear，{len(batches)} 批图像'
          f'（含真实帧={real.is_file()}）')

    # ---- 4) TensorRT 对照组（可选，必须在 MTQ 改模型之前）----
    trt_actions = None
    if args.trt:
        os.environ.setdefault('OPENPI_TRT_PRECISION', 'mtqfp8')
        os.environ.setdefault('OPENPI_FP8_SEGMENTS', 'vision,prefix,denoise')
        from deploy.tensorrt.runtime.fp8_policy import Fp8ActionSampler
        sampler = Fp8ActionSampler(model, artifact_dir=ARTIFACTS, device=args.device)
        trt_actions = []
        for obs, noise in observations:
            with torch.no_grad():
                trt_actions.append(_f32(sampler(args.device, obs, noise=noise.clone())[0]))
        print(f'[export] TensorRT mtqfp8 动作完成：vision={bool(sampler.vision)} '
              f'prefix={bool(sampler.prefix)} denoise={bool(sampler.denoise)}')
        del sampler

    # ---- 5) MTQ 校准 → Prefix/Denoise amax；再跑假量化动作（L1）----
    print('[export] mtq.quantize（Prefix + Denoise Linear，FP8 E4M3 逐张量）')
    amax['mtq'] = mtq_amax(model, calib, args.device, openpi_root)
    print(f'[export] MTQ amax：{len(amax["mtq"])} 个 Linear')
    fq_actions = []
    for obs, noise in observations:
        with torch.no_grad():
            fq_actions.append(_f32(model.sample_actions(args.device, obs, noise=noise.clone())[0]))

    amax['recipe'] = dict(
        format='fp8_e4m3', granularity='per-tensor', scale='amax/448',
        mtq='FP8_DEFAULT_CFG + run_step2._ALWAYS_DISABLE（Prefix 与 Denoise 的 Linear）',
        vision='forward hook max|x|，校准图像 = 16 条校准观测 + calib/pixel_values.npy',
    )
    (out_dir / 'fp8_amax.json').write_text(json.dumps(amax, indent=2) + '\n')

    # ---- 6) 写样本 ----
    rows = []
    for i, cap in enumerate(samples):
        cap['fq_actions'] = fq_actions[i]
        row = dict(sample=i, fq_rmse=_rmse(fq_actions[i], cap['ref_actions']))
        if trt_actions is not None:
            cap['trt_actions'] = trt_actions[i]
            row['trt_rmse'] = _rmse(trt_actions[i], cap['ref_actions'])
        np.savez(out_dir / 'samples' / f'sample_{i:03d}.npz', **cap)
        rows.append(row)
        print('[export] ' + '  '.join(f'{k}={v:.6f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))

    summary = dict(
        num_samples=args.num_samples, num_calib=args.num_calib, seed=args.seed, obs=args.obs,
        fq_rmse_mean=float(np.mean([r['fq_rmse'] for r in rows])),
        trt_rmse_mean=float(np.mean([r['trt_rmse'] for r in rows])) if trt_actions is not None else None,
        rows=rows, seconds=round(time.time() - t0, 1),
    )
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'[export] L1 MTQ 假量化 RMSE={summary["fq_rmse_mean"]:.6f}'
          + (f'  TensorRT RMSE={summary["trt_rmse_mean"]:.6f}' if trt_actions is not None else ''))
    share_outputs(out_dir)
    print(f'[export] 完成 {summary["seconds"]}s -> {out_dir}')


if __name__ == '__main__':
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
    main()
