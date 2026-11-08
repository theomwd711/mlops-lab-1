"""Quality gate for promoting a food11 model version.

Usage:
    uv run python src/food11/promote.py --version 8

Moves the `champion` alias to the candidate only if its test accuracy beats the
current champion by at least two standard errors; otherwise sets `challenger`.
The decision and both accuracies are written as tags on the candidate version.
"""

import argparse
import math
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "food11"
METRIC = "test_accuracy"
DATASET_DIRS = {"mini": "food11_processed_mini", "processed": "food11_processed"}


def run_of(client, version):
    """The training run behind a registered model version."""
    run_id = version.run_id
    if not run_id and version.source.startswith("models:/m-"):
        run_id = client.get_logged_model(version.source.removeprefix("models:/")).source_run_id
    if not run_id:
        raise SystemExit(f"cannot find the run behind {MODEL_NAME} v{version.version}")
    return client.get_run(run_id)


def accuracy_of(run):
    value = run.data.metrics.get(METRIC)
    if value is None:
        raise SystemExit(f"run {run.info.run_id} has no '{METRIC}' metric")
    return value


def test_set_size(run):
    dataset = run.data.params.get("dataset", "mini")
    folder = Path("data") / DATASET_DIRS[dataset] / "evaluation"
    return sum(1 for _ in folder.rglob("*.jpg"))


def gate(candidate_acc, champion_acc, n):
    """Candidate must beat the champion by 2 standard errors of the champion's accuracy."""
    se = math.sqrt(champion_acc * (1 - champion_acc) / n)
    threshold = champion_acc + 2 * se
    return candidate_acc >= threshold, se, threshold


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quality gate for promoting a food11 model version.")
    parser.add_argument("--version", required=True, type=int, help="candidate model version")
    parser.add_argument("--n", type=int, help="test set size (default: count the evaluation images)")
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()

    candidate = client.get_model_version(MODEL_NAME, str(args.version))
    candidate_acc = accuracy_of(run_of(client, candidate))

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        champion = None

    if champion is None:
        promoted = True
        tags = {"gate.reason": "no champion yet", "gate.candidate_accuracy": f"{candidate_acc:.4f}"}
        print(f"v{candidate.version}: no champion yet, promoting (accuracy {candidate_acc:.4f})")
    elif champion.version == candidate.version:
        print(f"v{candidate.version} is already the champion, nothing to do")
        return 0
    else:
        champion_run = run_of(client, champion)
        champion_acc = accuracy_of(champion_run)
        n = args.n or test_set_size(champion_run)
        if n == 0:
            raise SystemExit("found no evaluation images to size the test set; pass --n")
        promoted, se, threshold = gate(candidate_acc, champion_acc, n)
        tags = {
            "gate.candidate_accuracy": f"{candidate_acc:.4f}",
            "gate.champion_accuracy": f"{champion_acc:.4f}",
            "gate.champion_version": str(champion.version),
            "gate.test_set_size": str(n),
            "gate.threshold": f"{threshold:.4f}",
        }
        print(f"candidate v{candidate.version}: {candidate_acc:.4f}")
        print(f"champion  v{champion.version}: {champion_acc:.4f}  (n={n}, se={se:.4f}, 2*se={2 * se * 100:.2f} points)")
        print(f"needs >= {threshold:.4f} -> {'PROMOTE' if promoted else 'REJECT'}")

    tags["gate.decision"] = "promoted" if promoted else "rejected"
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, str(candidate.version), key, value)

    alias = "champion" if promoted else "challenger"
    client.set_registered_model_alias(MODEL_NAME, alias, str(candidate.version))
    print(f"alias '{alias}' -> v{candidate.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())