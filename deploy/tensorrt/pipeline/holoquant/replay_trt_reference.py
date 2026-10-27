#!/usr/bin/env python3
"""Replay existing pi05 exports through real TRT engines without changing exports.

Use OpenPI's Python environment. All three engines must load. Saved inputs and
noise are reused, and L0 is replayed to check reconstruction of each observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from deploy.tensorrt.common import ARTIFACTS, load_pi05_pytorch
from deploy.tensorrt.runtime.fp8_policy import Fp8ActionSampler
from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(a):
    a = np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()[:16]


def metric(ref, hat):
    if ref.shape != hat.shape or not np.isfinite(ref).all() or not np.isfinite(hat).all():
        raise ValueError('Invalid action shape or nonfinite actions')
    ref, hat = ref.astype(np.float64).reshape(-1), hat.astype(np.float64).reshape(-1)
    error = hat - ref
    denom = np.linalg.norm(ref) * np.linalg.norm(hat)
    if denom == 0:
        raise ValueError('Undefined cosine for zero norm')
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))),
                max_abs=float(np.max(np.abs(error))), cosine=float(np.dot(ref, hat) / denom))


def summarize(rows, key):
    values = [r[key] for r in rows]
    result = {k: float(np.mean([r[k] for r in values])) for k in values[0]}
    result['rmse_max'] = max(r['rmse'] for r in values)
    result['cosine_min'] = min(r['cosine'] for r in values)
    result['global_max_abs'] = max(r['max_abs'] for r in values)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--openpi-out', type=Path, required=True)
    ap.add_argument('--l2-run-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--artifact-dir', type=Path, default=ARTIFACTS)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-samples', type=int)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU or fake-quant fallback')
    if args.max_samples is not None and args.max_samples <= 0:
        ap.error('--max-samples must be positive')
    export, run = args.openpi_out.resolve(), args.l2_run_dir.resolve()
    source_paths = [export / n for n in ('config.json', 'fp8_amax.json', 'summary.json')]
    source_paths += [run / 'l2_ba2701.json', run / 'actions_ba2701.npz']
    source_shas = {str(p): file_sha(p) for p in source_paths}
    summary = json.loads((export / 'summary.json').read_text())
    cfg = json.loads((export / 'config.json').read_text())
    l2 = json.loads((run / 'l2_ba2701.json').read_text())
    names = l2['sample_files']
    exported = sorted(p.name for p in (export / 'samples').glob('sample_*.npz'))
    if names != exported or len(names) != len(set(names)) or len(names) != len(l2['rows']):
        raise ValueError('L2/export sample manifest mismatch')
    if l2['partial_graph'] or l2['from_ref_prefix'] or l2['from_ref_suffix']:
        raise ValueError('Expected a complete L2 rollout')
    for fp in l2['fingerprint'].values():
        if fp != {'fp8_amax': source_shas[str(export / 'fp8_amax.json')][:16],
                  'config': source_shas[str(export / 'config.json')][:16]}:
            raise ValueError('L2/export configuration mismatch')
    checkpoint = Path(summary['checkpoint']['path'])
    checkpoint_sha = file_sha(checkpoint)
    if checkpoint_sha != summary['checkpoint']['sha256'] or checkpoint_sha != l2['checkpoint']['sha256']:
        raise ValueError('Checkpoint SHA mismatch')
    os.environ['OPENPI_TRT_PRECISION'] = 'mtqfp8'
    os.environ['OPENPI_FP8_SEGMENTS'] = 'vision,prefix,denoise'
    engines = {s: args.artifact_dir / 'engines' / f'{s}_mtqfp8.plan'
               for s in ('vision', 'prefix', 'denoise')}
    engine_info = {s: dict(path=str(p), resolved_path=str(p.resolve()), sha256=file_sha(p))
                   for s, p in engines.items()}
    if args.out_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out_dir}')
    args.out_dir.mkdir(parents=True)
    started = time.time()
    print('[replay] Loading the matching BF16/FP32 model and all three TRT engines', flush=True)
    model, _ = load_pi05_pytorch(checkpoint.parent, args.device)
    if not model.pi05 or (model.config.action_horizon, model.config.action_dim) != (
            cfg['action_horizon'], cfg['action_dim']):
        raise ValueError('Saved-state replay supports matching pi05 only')
    sampler = Fp8ActionSampler(model, artifact_dir=args.artifact_dir, device=args.device)
    for segment in engines:
        runner = getattr(sampler, segment)
        if runner is None:
            raise RuntimeError(f'TRT engine missing: {segment}; fallback is forbidden')
        engine_info[segment]['io'] = {
            n: dict(shape=list(runner.engine.get_tensor_shape(n)),
                    dtype=str(runner.engine.get_tensor_dtype(n)))
            for n in runner.input_names + runner.output_names}
    print('[replay] All TRT engines loaded; preprocessing and L0 will be checked per sample', flush=True)
    records, trt_arrays, l0_arrays = [], {}, {}
    rows_by_name = {r['sample_file']: r for r in l2['rows']}
    selected = names if args.max_samples is None else names[:args.max_samples]
    l2_actions = np.load(run / 'actions_ba2701.npz', allow_pickle=False)
    for i, name in enumerate(selected):
        path = export / 'samples' / name
        original_sha = file_sha(path)
        with np.load(path, allow_pickle=False) as z:
            inputs = {k: z[k].copy() for k in ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'noise')}
            ref = z['ref_actions'].copy()
            meta = {k: z[k].item() for k in ('suite', 'task_id', 'episode')}
        if array_sha(ref) != rows_by_name[name]['ref_sha']:
            raise ValueError(f'Reference action identity mismatch: {name}')
        tensor = lambda a: torch.from_numpy(np.asarray(a).copy()).to(args.device)
        images = [tensor(a[None]) for a in inputs['images']]
        masks = [tensor(np.array([m], dtype=bool)) for m in inputs['img_masks']]
        # pi05 encodes state in the saved prompt; continuous state is not used by embed_suffix.
        obs = SimpleNamespace(images=dict(zip(IMAGE_KEYS, images, strict=True)),
                              image_masks=dict(zip(IMAGE_KEYS, masks, strict=True)),
                              tokenized_prompt=tensor(inputs['lang_tokens'][None]),
                              tokenized_prompt_mask=tensor(inputs['lang_masks'][None]),
                              state=torch.zeros((1, cfg['action_dim']), device=args.device),
                              token_ar_mask=None, token_loss_mask=None)
        processed = model._preprocess_observation(obs, train=False)
        expected = (images, masks, obs.tokenized_prompt, obs.tokenized_prompt_mask)
        for actual, original in zip(processed[:4], expected, strict=True):
            aa, bb = (actual, original) if isinstance(original, list) else ([actual], [original])
            if not all(torch.equal(a, b) for a, b in zip(aa, bb, strict=True)):
                raise ValueError('Reconstructed preprocessing changes saved inputs')
        noise = tensor(inputs['noise'][None])
        with torch.no_grad():
            l0 = model.sample_actions(args.device, obs, noise=noise.clone(), num_steps=cfg['num_steps'])
            actual = sampler(args.device, obs, noise=noise.clone(), num_steps=cfg['num_steps'])
        l0_np, trt_np = l0[0].float().cpu().numpy(), actual[0].float().cpu().numpy()
        replay_error = metric(ref, l0_np)
        # An explicit tight gate guards against mismatched inputs or model revisions.
        if replay_error['max_abs'] > 1e-5:
            raise ValueError(f'L0 replay mismatch for {name}: {replay_error}')
        key = Path(name).stem
        l2_np = l2_actions[key][0]
        row = dict(sample=key, **meta, ref_sha=array_sha(ref), sample_file_sha256=original_sha,
                   input_array_hashes={k: array_sha(v) for k, v in inputs.items()},
                   l0_replay_vs_saved=replay_error,
                   l0_replay_bitwise=bool(l0_np.dtype == ref.dtype and l0_np.tobytes() == ref.tobytes()),
                   trt_vs_l0=metric(ref, trt_np), l2_vs_l0=metric(ref, l2_np),
                   l2_vs_trt=metric(trt_np, l2_np),
                   trt_vs_l0_active7=metric(ref[:, :7], trt_np[:, :7]),
                   l2_vs_trt_active7=metric(trt_np[:, :7], l2_np[:, :7]))
        if file_sha(path) != original_sha:
            raise RuntimeError('Input file changed during replay')
        records.append(row)
        trt_arrays[key], l0_arrays[key] = actual.float().cpu().numpy(), l0.float().cpu().numpy()
        print(f"[replay] {i+1}/{len(selected)} {key} TRT/L0 RMSE={row['trt_vs_l0']['rmse']:.9f} "
              f"cos={row['trt_vs_l0']['cosine']:.9f} L2/TRT RMSE={row['l2_vs_trt']['rmse']:.9f} "
              f"L0 bitwise={row['l0_replay_bitwise']}", flush=True)
    l2_actions.close()
    for path, digest in source_shas.items():
        if file_sha(path) != digest:
            raise RuntimeError(f'Source file changed: {path}')
    for segment, path in engines.items():
        if file_sha(path) != engine_info[segment]['sha256']:
            raise RuntimeError(f'Engine changed: {segment}')
    metric_keys = ('trt_vs_l0', 'l2_vs_l0', 'l2_vs_trt', 'trt_vs_l0_active7', 'l2_vs_trt_active7')
    import tensorrt
    output = dict(n_samples=len(records), n_steps=cfg['num_steps'], complete=selected == names,
                  metric='float64 metrics per sample, then arithmetic mean; model-space 10x32 outputs',
                  source_files=source_shas, checkpoint_sha256=checkpoint_sha, engines=engine_info,
                  engine_build_checkpoint_and_amax_verified=False,
                  comparison_scope='Actual deployment pipeline vs saved L0/L2, not isolated hardware/quantization error',
                  caveats=['Engine build-time checkpoint/calibration binding is not established by file names.',
                           'Vision uses the existing vision_fp8.plan via vision_mtqfp8.plan symlink.',
                           'Suffix embedding, action projection and Euler updates stay in PyTorch.'],
                  environment=dict(python=sys.version, torch=torch.__version__, tensorrt=tensorrt.__version__,
                                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(args.device),
                                   visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
                  l0_replay_all_bitwise=all(r['l0_replay_bitwise'] for r in records),
                  l0_replay_max_abs=max(r['l0_replay_vs_saved']['max_abs'] for r in records),
                  summary={k: summarize(records, k) for k in metric_keys},
                  suites={s: {k: summarize([r for r in records if r['suite'] == s], k) for k in metric_keys}
                          for s in dict.fromkeys(r['suite'] for r in records)},
                  rows=records, seconds=time.time() - started,
                  code_sha256={str(p): file_sha(p) for p in (Path(__file__),
                      ROOT / 'deploy/tensorrt/runtime/fp8_policy.py',
                      ROOT / 'deploy/tensorrt/runtime/trt_runner.py',
                      ROOT / 'src/openpi/models_pytorch/pi0_pytorch.py',
                      ROOT / 'deploy/tensorrt/common.py')})
    np.savez(args.out_dir / 'actions_trt.npz', **trt_arrays)
    np.savez(args.out_dir / 'actions_l0_replay.npz', **l0_arrays)
    (args.out_dir / 'trt_replay.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print('[replay] Saved', args.out_dir, json.dumps(output['summary']), flush=True)


if __name__ == '__main__':
    main()
