# Lab 2 — Model Training and Experiment Tracking with MLflow

Answers below are grounded in what actually happened in this repo — the real
`pyproject.toml`/`uv.lock` diff, the actual `.gitignore` bug this project
hit, and what we later confirmed on disk in Lab 3 while debugging model
loading. Q7–Q9 need your own numbers from the mlflow UI (noted where).

## Q1 — What changed in `pyproject.toml`/`uv.lock`

`pyproject.toml` gained:

```toml
dependencies = [
    ...
    "mlflow>=3.16.0",
    "torch>=2.14.0",
    "torchvision>=0.29.0",
    "scikit-learn>=1.9.1",
]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cuda" }
torchvision = { index = "pytorch-cuda" }
```

The `[[tool.uv.index]]` block declares a second package index (PyTorch's own
CUDA 12.6 wheel index) alongside PyPI; `explicit = true` means uv only routes
packages there if something in `[tool.uv.sources]` says so — it doesn't
become a general fallback for everything. The `[tool.uv.sources]` block does
that routing for `torch`/`torchvision` specifically, so those two resolve to
`+cu126` wheels instead of PyPI's default (CPU-only on Windows).

`uv.lock` grew by well over a thousand lines: torch/torchvision now pin to
their `+cu126` build strings, and a long tail of `nvidia-*-cu12` packages
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cuda-cupti-cu12`,
`nvidia-cuda-nvrtc-cu12`, `nvidia-cufft-cu12`, `nvidia-curand-cu12`,
`nvidia-cusolver-cu12`, `nvidia-cusparse-cu12`, `nvidia-cusparselt-cu12`,
`nvidia-nccl-cu12`, `nvidia-nvjitlink-cu12`, `nvidia-nvtx-cu12`,
`nvidia-nvshmem-cu12`, plus `cuda-bindings` and `triton`) got pulled in as
torch's own runtime dependencies — several GB combined. This came back to
bite us hard in Lab 3, when building a CPU-only serving container: excluding
`torch`/`torchvision` by name from the install did **not** exclude these —
they're independent entries in the lock, not implicitly tied to torch's
presence (see Lab 3 notes).

## Q2 — `--backend-store-uri` vs. `--default-artifact-root`

- **`--backend-store-uri sqlite:///mlflow.db`** — where mlflow stores its own
  *metadata*: experiments, runs, params, metrics, tags, and registered
  models/versions/aliases. Small, structured, queryable — hence a SQL
  backend.
- **`--default-artifact-root ./mlruns`** — where mlflow stores *artifacts*:
  the actual large binary files a run produces (model weights, plots, etc.).
  Unstructured, potentially huge — not something you'd want as a database
  row.

Concretely in this project: a run's `val_accuracy` numbers live in
`mlflow.db`; the actual trained model file lives under `./mlruns/...` (see
Q6) — two different stores for two fundamentally different kinds of data.

## Q3 — Why `mlflow.db`/`mlruns/` shouldn't be tracked by git or dvc

Both are pure local **run output**, regenerated every time you train —
not source code and not a deliberately-versioned dataset. Same logic as why
`data/` is gitignored (Lab 1 Q4): derived/generated output doesn't belong in
git history, where it would balloon with every single run.

dvc shouldn't track them either, for a different reason: dvc's model is
"version this specific, deliberately-pinned snapshot of data" — but mlflow's
tracking store *is already* a versioning system for runs/metrics; that's its
entire purpose. Layering dvc on top would be redundant, and would actively
conflict with mlflow's own bookkeeping — mlflow expects to own and
continuously rewrite `mlflow.db`/`mlruns` (new runs, new metrics, new
aliases), while dvc's model is built around immutable, content-addressed
snapshots. The two aren't compatible for the same files.

**A real gotcha we hit in this exact repo**: the `.gitignore` lines for
these two paths were originally added via PowerShell's
`echo "mlflow.db" >> .gitignore` — which silently writes in UTF-16 while the
rest of the file was UTF-8, corrupting it (git even displayed the diff as
binary: `.gitignore | Bin 13 -> 55 bytes`). The result: `mlflow.db` and
`mlruns/` were listed in `.gitignore` but **not actually ignored** —
confirmed with `git check-ignore`, which returned no match. Fixed in a
follow-up commit ("Fix .gitignore encoding") by rewriting the file in plain
UTF-8. Lesson: "it's in `.gitignore`" and "it's actually ignored" are not
guaranteed to be the same thing — worth verifying with `git check-ignore
-v <path>` after editing `.gitignore` from PowerShell.

## Q4 — First `set_experiment` call with a new name

It creates a brand-new experiment in the tracking store immediately — it
gets its own experiment ID and appears right away in the mlflow UI's left
sidebar, with zero runs under it until you actually log one. Every
subsequent call to `mlflow.start_run()` in that process logs under whichever
experiment the most recent `set_experiment` call selected.

## Q5 — `log_param` vs. `log_metric`, and why only one takes `step`

- **`log_param`** — a value fixed *before* training starts and never
  changes for the life of the run (learning rate, batch size, dataset
  choice, model architecture name). Logged once. There's no meaningful
  notion of "this parameter at epoch 3" — it's a single fact about the run's
  configuration.
- **`log_metric`** — a value produced *during or after* training that can
  legitimately have many values over the run's lifetime (`train_loss`,
  `val_loss`, `val_accuracy` — one value per epoch here). `step` (this
  project uses `step=epoch`) is what turns that into a proper time series
  mlflow can plot as a line chart instead of overwriting a single number —
  it's literally the x-axis of the metric chart.

## Q6 — Where the model artifact actually lives on disk

This one has a real, verified answer rather than the "textbook" one, because
we had to track it down precisely in Lab 3 to make container-based serving
work at all. With mlflow 3.16 (this project's version), `mlflow.pytorch.
log_model(...)` doesn't just attach the model to the run's own artifact
folder the way older mlflow tutorials describe — it creates a separate
**Logged Model** entity with its own ID (e.g. `m-758384aeebe64a24b4a1a
8176421f394`), and the actual files land at:

```
mlruns/<experiment_id>/models/m-<logged_model_id>/artifacts/
```

— not the older `mlruns/<experiment_id>/<run_id>/artifacts/model/` layout.
Confirmed directly against this repo via:

```python
client.get_logged_model("m-758384aeebe64a24b4a1a8176421f394").artifact_location
# -> 'file:C:/Users/96171/PycharmProjects/mlops-lab-1/mlruns/1/models/m-758384aeebe64a24b4a1a8176421f394/artifacts'
```

This distinction mattered a lot in Lab 3 — see the Lab 3 write-up for why a
raw local-disk artifact location like this one doesn't travel well into a
Docker container.

## Q7 — Best learning rate; is higher always better

The four sweep runs actually trained in this project (5 epochs, mini
dataset, batch configs `lr=0.01/bs=32`, `lr=0.001/bs=32`, `lr=0.0001/bs=32`,
`lr=0.001/bs=64` — named `illustrious-ram-146`, `amazing-quail-356`,
`skillful-croc-910`, `casual-gull-469` in the mlflow UI) were used to pick a
champion run (`d2df6817d90e47ecbba4b71cdaf0f70a`, later registered as
`food11` version 1). **I don't have the actual per-run `val_accuracy`
numbers** to state which learning rate won or whether higher was better —
that needs to come from your own mlflow UI. Open the `food11` experiment,
sort the runs table by `val_accuracy` descending, and note: which `lr` is on
top, and whether `lr=0.01` (highest) actually beat the smaller learning
rates or over/undershot.

## Q8 — Parallel coordinates pattern (lr, batch_size, val_accuracy)

Same caveat as Q7 — fill this in from the actual Compare page. What to look
for specifically: whether the best `val_accuracy` line passes through a
*middle* `lr` value rather than an extreme one (a common pattern — too high
diverges, too low undertrains in 5 epochs), and whether `batch_size=64`
(only tested at `lr=0.001`) looks meaningfully different from
`batch_size=32` at the same learning rate, or within noise given only 5
epochs and a 100-image-per-category mini dataset.

## Q9 — Best run by `val_accuracy`

The run registered as `food11` version 1 / `champion` alias was
`d2df6817d90e47ecbba4b71cdaf0f70a` — presumably chosen as the best run on
`val_accuracy` per the lab's own instructions ("sort the runs table ...,
open your best run"), though I don't have direct confirmation of the numeric
value that made it the winner. Worth double-checking in the UI that this is
genuinely the top row after sorting, not a leftover debug run — this project
explicitly ran into that risk (early debugging reruns while fixing the
`mlflow.pytorch.log_model` pickle-format bug all landed at
`lr=0.001, batch_size=32`, indistinguishable from the real sweep entry at a
glance).
