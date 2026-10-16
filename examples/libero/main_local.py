"""LIBERO evaluation with in-process openpi bf16 inference (no policy server).

This is the same rollout loop as examples/libero/main.py, but loads the model
locally via create_trained_policy() instead of WebsocketClientPolicy.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys

# Ensure third_party LIBERO is importable even if PYTHONPATH was not set.
_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LIBERO_ROOT = _OPENPI_ROOT / "third_party" / "libero"
if _LIBERO_ROOT.is_dir() and str(_LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIBERO_ROOT))

# /tmp/torchinductor_$USER can be owned by another account on a shared machine.
# torch.compile (max-autotune) must be able to write its kernel cache.
_inductor_cache = pathlib.Path.home() / ".cache" / "torchinductor"
_inductor_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_inductor_cache))

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import torch
import tqdm
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# PyTorch>=2.6 defaults torch.load(weights_only=True); LIBERO init-state pickles need False.
_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat  # type: ignore[assignment]

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Local openpi checkpoint (bf16 JAX params loaded as bfloat16).
    config_name: str = "pi05_libero"
    checkpoint_dir: str = "/share_data/public/models/openpi/pi05_libero"

    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"  # libero_spatial/object/goal/10/90
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    # If >0, only evaluate the first N tasks (useful for smoke tests).
    max_tasks: int = 0

    video_out_path: str = "data/libero/videos_local"
    seed: int = 7
    # Multi-GPU data parallel: each process handles jobs where index % num_shards == shard_id.
    shard_id: int = 0
    num_shards: int = 1
    results_json: str = ""
    # auto: jax vs pytorch from checkpoint files. fp8: TensorRT engines + Python leftovers.
    backend: str = "auto"
    fp8_artifact_dir: str = ""
    # Which TensorRT segments to use. Override with env OPENPI_FP8_SEGMENTS.
    # Production path: OPENPI_TRT_PRECISION=mtqfp8 and segments vision,prefix,denoise
    # (all-three RMSE ~0.011, LIBERO avg 96.95%). The old ONNX-PTQ prefix/denoise
    # engines (~0.25 RMSE) must not be used.


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    logging.info(
        "Loading local policy: config=%s dir=%s backend=%s",
        args.config_name,
        args.checkpoint_dir,
        args.backend,
    )
    train_cfg = _config.get_config(args.config_name)
    # torch.compile/Triton tries to gcc-link -lcuda. That is not needed to check
    # JAX→PyTorch conversion (or later ONNX export), and this machine has no
    # unsuffixed libcuda.so on the default linker path.
    object.__setattr__(train_cfg.model, "pytorch_compile_mode", None)
    if args.backend == "fp8":
        if str(_OPENPI_ROOT) not in sys.path:
            sys.path.insert(0, str(_OPENPI_ROOT))
        from deploy.tensorrt.runtime.fp8_policy import create_fp8_policy

        artifact = args.fp8_artifact_dir or str(_OPENPI_ROOT / "deploy" / "tensorrt" / "artifacts")
        logging.info("FP8 artifacts: %s", artifact)
        policy = create_fp8_policy(train_cfg, args.checkpoint_dir, artifact)
    else:
        policy = _policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint_dir,
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.max_tasks and args.max_tasks > 0:
        num_tasks_in_suite = min(num_tasks_in_suite, args.max_tasks)
    logging.info(
        "Task suite: %s (%d tasks) shard %d/%d",
        args.task_suite_name,
        num_tasks_in_suite,
        args.shard_id,
        args.num_shards,
    )
    if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"invalid shard: id={args.shard_id} num={args.num_shards}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    all_jobs = [
        (task_id, episode_idx)
        for task_id in range(num_tasks_in_suite)
        for episode_idx in range(args.num_trials_per_task)
    ]
    my_jobs = [job for i, job in enumerate(all_jobs) if i % args.num_shards == args.shard_id]
    jobs_by_task: dict[int, list[int]] = collections.defaultdict(list)
    for task_id, episode_idx in my_jobs:
        jobs_by_task[task_id].append(episode_idx)
    logging.info("This shard will run %d / %d episodes", len(my_jobs), len(all_jobs))

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(sorted(jobs_by_task)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(jobs_by_task[task_id]):
            logging.info("\nTask: %s", task_description)
            env.reset()
            action_plan: collections.deque = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            logging.info("Starting episode %d...", task_episodes + 1)
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = policy.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"Need >= {args.replan_steps} actions, got {len(action_chunk)}"
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_t{task_id:02d}_e{episode_idx:03d}_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %s", total_episodes)
            logging.info(
                "# successes: %s (%.1f%%)",
                total_successes,
                total_successes / total_episodes * 100,
            )

        logging.info("Current task success rate: %s", float(task_successes) / float(task_episodes))
        logging.info("Current total success rate: %s", float(total_successes) / float(total_episodes))
        try:
            env.close()
        except Exception:
            pass

    if total_episodes:
        logging.info("Total success rate: %s", float(total_successes) / float(total_episodes))
    logging.info("Total episodes: %s", total_episodes)
    if args.results_json:
        out = {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "task_suite_name": args.task_suite_name,
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": (float(total_successes) / float(total_episodes) if total_episodes else 0.0),
        }
        path = pathlib.Path(args.results_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        logging.info("Wrote shard results to %s", path)


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(eval_libero)
