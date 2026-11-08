"""Validates a processed Food-11 dataset against the data contract.

Usage:
    uv run python src/food11/validate_data.py data/food11_processed_mini [--min-per-class 50] [--tau 0.05]

Prints one PASS/FAIL line per check and exits with status 1 if any check fails.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

try:
    from src.food11.data import CLASSES
except ModuleNotFoundError:  # run as a script: python src/food11/validate_data.py
    from data import CLASSES

SPLITS = ("training", "validation", "evaluation")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (128, 128)

Result = tuple[str, bool, str]
Dataset = dict[str, dict[str, list[Path]]]


def load_dataset(root: Path) -> Dataset:
    """Maps split -> class folder name -> image paths."""
    dataset: Dataset = {}
    for split in SPLITS:
        split_dir = root / split
        classes: dict[str, list[Path]] = {}
        if split_dir.is_dir():
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                classes[class_dir.name] = sorted(
                    p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                )
        dataset[split] = classes
    return dataset


def short(items: list[str], limit: int = 5) -> str:
    text = "; ".join(items[:limit])
    return text + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def check_class_folders(dataset: Dataset) -> Result:
    problems = []
    for split in SPLITS:
        found = set(dataset[split])
        if found != set(CLASSES):
            missing = sorted(set(CLASSES) - found)
            extra = sorted(found - set(CLASSES))
            problems.append(f"{split}: missing {missing}, unexpected {extra}")
    return ("11 class folders in every split", not problems, short(problems) or "ok")


def check_images(dataset: Dataset) -> Result:
    bad = []
    total = 0
    for split in SPLITS:
        for paths in dataset[split].values():
            for path in paths:
                total += 1
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB" or img.size != IMAGE_SIZE:
                            bad.append(f"{path} is {img.mode} {img.size[0]}x{img.size[1]}")
                except Exception as exc:
                    bad.append(f"{path} cannot be read ({exc})")
    return ("images are RGB 128x128", not bad, short(bad) or f"{total} images checked")


def check_min_per_class(dataset: Dataset, minimum: int) -> Result:
    low = [
        f"{split}/{name}: {len(dataset[split].get(name, []))}"
        for split in SPLITS
        for name in CLASSES
        if len(dataset[split].get(name, [])) < minimum
    ]
    return (f"at least {minimum} images per class", not low, short(low) or "ok")


def check_no_leakage(dataset: Dataset) -> Result:
    first_seen: dict[str, tuple[str, Path]] = {}
    leaks = []
    for split in SPLITS:
        for paths in dataset[split].values():
            for path in paths:
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                other_split, other_path = first_seen.setdefault(digest, (split, path))
                if other_split != split:
                    leaks.append(f"{path} duplicates {other_path}")
    return ("no image shared between splits", not leaks, short(leaks) or "ok")


def check_class_balance(dataset: Dataset, tau: float) -> Result:
    name = f"class share train vs validation within {tau}"
    train_total = sum(len(p) for p in dataset["training"].values())
    val_total = sum(len(p) for p in dataset["validation"].values())
    if not train_total or not val_total:
        return (name, False, "training or validation split is empty")
    drifts = []
    for cls in CLASSES:
        train_share = len(dataset["training"].get(cls, [])) / train_total
        val_share = len(dataset["validation"].get(cls, [])) / val_total
        if abs(train_share - val_share) > tau:
            drifts.append(f"{cls}: {train_share:.1%} train vs {val_share:.1%} validation")
    return (name, not drifts, short(drifts) or "ok")


def run_checks(root: Path, min_per_class: int = 50, tau: float = 0.05) -> list[Result]:
    dataset = load_dataset(root)
    return [
        check_class_folders(dataset),
        check_images(dataset),
        check_min_per_class(dataset, min_per_class),
        check_no_leakage(dataset),
        check_class_balance(dataset, tau),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a processed Food-11 dataset.")
    parser.add_argument("root", type=Path, help="e.g. data/food11_processed_mini")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args(argv)

    results = run_checks(args.root, args.min_per_class, args.tau)
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    return 0 if all(passed for _, passed, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())