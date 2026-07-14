#!/usr/bin/env python3
"""Download common 3DGS research datasets from Hugging Face.

The resulting paths are arranged to match this repository's examples:

  data/nerf_synthetic/lego
  data/mipnerf360/bonsai
  data/mipnerf360/garden
  data/mipnerf360/bicycle
  data/tandt/{train,truck}
  data/db/{drjohnson,playroom}
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetTarget:
    name: str
    repo_id: str
    local_dir: str
    patterns: tuple[str, ...]
    expected_paths: tuple[str, ...]


TARGETS: dict[str, DatasetTarget] = {
    "lego": DatasetTarget(
        name="lego",
        repo_id="rishitdagli/nerf-gs-datasets",
        local_dir="nerf_synthetic",
        patterns=("lego/**",),
        expected_paths=("nerf_synthetic/lego/transforms_train.json",),
    ),
    "bonsai": DatasetTarget(
        name="bonsai",
        repo_id="alexmkwizu/gaussian_training_datasets",
        local_dir=".",
        patterns=("mipnerf360/bonsai/**",),
        expected_paths=("mipnerf360/bonsai/images", "mipnerf360/bonsai/sparse/0"),
    ),
    "garden": DatasetTarget(
        name="garden",
        repo_id="alexmkwizu/gaussian_training_datasets",
        local_dir=".",
        patterns=("mipnerf360/garden/**",),
        expected_paths=("mipnerf360/garden/images", "mipnerf360/garden/sparse/0"),
    ),
    "bicycle": DatasetTarget(
        name="bicycle",
        repo_id="alexmkwizu/gaussian_training_datasets",
        local_dir=".",
        patterns=("mipnerf360/bicycle/**",),
        expected_paths=("mipnerf360/bicycle/images", "mipnerf360/bicycle/sparse/0"),
    ),
    "tandt_db": DatasetTarget(
        name="tandt_db",
        repo_id="alexmkwizu/gaussian_training_datasets",
        local_dir=".",
        patterns=("tandt/train/**", "tandt/truck/**", "db/drjohnson/**", "db/playroom/**"),
        expected_paths=("tandt/train/sparse/0", "tandt/truck/sparse/0", "db/drjohnson/sparse/0", "db/playroom/sparse/0"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        help=f"Datasets to download. Choices: {', '.join(sorted(TARGETS.keys()))}. Default: all configured targets.",
    )
    parser.add_argument("--data-dir", default="data", help="Output data directory. Default: data")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--revision", default="main", help="Dataset repository revision. Default: main")
    parser.add_argument("--token", default=None, help="Hugging Face token, or set HF_TOKEN.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading.")
    return parser.parse_args()


def download_target(target: DatasetTarget, data_dir: Path, cache_dir: str | None, revision: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    local_dir = (data_dir / target.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n== {target.name} ==")
    print(f"repo: {target.repo_id}")
    print(f"include: {', '.join(target.patterns)}")
    print(f"local_dir: {local_dir}")

    snapshot_download(
        repo_id=target.repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(target.patterns),
        local_dir=str(local_dir),
        cache_dir=cache_dir,
        token=token,
        local_dir_use_symlinks=False,
    )

    for relative_path in target.expected_paths:
        expected = data_dir / relative_path
        if not expected.exists():
            raise FileNotFoundError(f"Expected path was not created: {expected}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    token = args.token or os.environ.get("HF_TOKEN")

    print(f"data_dir: {data_dir}")
    dataset_names = args.datasets or ["lego", "bonsai", "garden", "bicycle", "tandt_db"]
    unknown = sorted(set(dataset_names) - set(TARGETS))
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Choices: {', '.join(sorted(TARGETS))}")

    for dataset_name in dataset_names:
        target = TARGETS[dataset_name]
        if args.dry_run:
            print(f"{dataset_name}: {target.repo_id} -> {data_dir / target.local_dir} ({target.patterns})")
            continue
        download_target(target, data_dir, args.cache_dir, args.revision, token)

    print("\nDone.")


if __name__ == "__main__":
    main()
