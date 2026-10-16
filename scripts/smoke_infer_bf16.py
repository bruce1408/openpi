#!/usr/bin/env python3
"""Smoke-test openpi bf16 inference with a local checkpoint (no robot needed)."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        default="/share_data/public/models/openpi/pi05_droid",
        help="Local openpi checkpoint directory",
    )
    parser.add_argument("--config_name", default="pi05_droid")
    parser.add_argument("--prompt", default="pick up the fork")
    args = parser.parse_args()

    print("jax.devices:", jax.devices())
    print("default_backend:", jax.default_backend())

    train_config = _config.get_config(args.config_name)
    print(f"Loading policy config={args.config_name} from {args.checkpoint_dir}")
    t0 = time.time()
    policy = policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    print(f"Policy loaded in {time.time() - t0:.1f}s")

    # Inspect parameter dtypes after load (bf16 expected for most).
    model = policy._model  # noqa: SLF001 - smoke test only
    leaves = jax.tree_util.tree_leaves(model)
    dtypes: dict[str, int] = {}
    for leaf in leaves:
        if hasattr(leaf, "dtype"):
            key = str(leaf.dtype)
            dtypes[key] = dtypes.get(key, 0) + 1
    print("model array dtype leaf counts:", dtypes)

    example = droid_policy.make_droid_example()
    example["prompt"] = args.prompt

    # Warmup + timed infer
    out = policy.infer(example)
    t1 = time.time()
    out = policy.infer(example)
    dt_ms = (time.time() - t1) * 1000

    actions = out["actions"]
    print(f"actions shape={np.asarray(actions).shape} dtype={np.asarray(actions).dtype}")
    print(f"actions[0,:8]={np.asarray(actions)[0, :8]}")
    print(f"infer latency (2nd call): {dt_ms:.1f} ms")
    print("OK: bf16 openpi inference succeeded")


if __name__ == "__main__":
    main()
