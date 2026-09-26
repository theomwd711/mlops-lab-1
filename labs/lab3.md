# Lab 3 — Containerizing the Model with Docker

Answers grounded in the actual merged `Dockerfile`, `.dockerignore`,
`src/food11/serve.py`, and everything we actually hit and fixed while
getting this container to run. Q5 and Q8 need commands run on your machine
— flagged below rather than invented.

## Q1 — Model version, and run artifact vs. registered model

**Version: `1`.** Confirmed directly: `Created version '1' of model
'food11'.`, then `client.set_registered_model_alias('food11', 'champion',
1)`.

A run's logged model artifact (`mlflow.pytorch.log_model(...)` inside
`with mlflow.start_run()`) is tied to that one specific training execution
— an immutable byproduct of that run, referenced as a "Logged Model"
(`models:/m-<id>`, per Lab 2 Q6) or historically as `runs:/<run_id>/model`.
A **registered model** (`mlflow.register_model(...)` under the name
`food11`) is a separate naming/versioning layer on top of that: it gives
the artifact a stable name (`food11`) and version number (`1`, `2`, ...)
that exists independently of which run produced it, so downstream code can
reference "the model" by name+version without knowing or caring which
run_id it came from.

## Q2 — Aliases vs. deprecated stages

Aliases (`champion`, `challenger`, etc.) replaced the old built-in
`Staging`/`Production` stage names.

Versioning separately from the run matters because a run captures one
specific training execution (data, code, hyperparams, environment at that
moment), while a registered model version is a stable, citable reference to
"the model currently deployed as X" — consumers shouldn't need to know or
care which run produced it, and you want the freedom to point that name at
a *different* underlying run later without renaming anything downstream.

An alias is more flexible than a fixed stage name because stages were a
small, hardcoded enum (`Staging`/`Production`/`Archived`) shared globally
across every registered model in the instance, whereas aliases are
free-form strings you invent yourself, can have as many of as you want
(`champion`, `challenger`, `shadow`, `canary-5pct`...), and reassigning one
to a new version is just a metadata update — it doesn't touch or rename the
underlying artifact, so you can promote a new version to `champion`
instantly and atomically, with the previous version still fully intact
under its own version number if you need to roll back.

## Q3 — Loading via `models:/food11@champion` instead of the `.pth` file

Loading via the model URI keeps `serve.py` completely decoupled from
*which* physical file/run/version is actually behind "the model currently in
production" — the alias is the only thing it needs to know. It also gets
mlflow's own loading machinery for free: flavor detection, dependency
mismatch warnings (we saw these directly at container startup — a big
`mlflow.utils.requirements_utils` warning listing every package/version
difference between the training environment and the serving container), and
consistent behavior across any pyfunc-compatible model type, instead of
hand-rolling `torch.load` plus custom pre/post-processing.

To serve a newer version, `serve.py` and the `Dockerfile` don't change at
all — you just repoint the alias:
```python
client.set_registered_model_alias("food11", "champion", <new_version_number>)
```
then restart the container. The next `mlflow.pyfunc.load_model(...)` call at
startup resolves to the new version automatically.

## Q4 — Why `pyproject.toml`/`uv.lock` copy + `uv sync` before the rest of the source

Docker builds one layer per instruction and caches each independently; a
layer's cache is invalidated (forcing it and every layer *after* it to
re-run) only when that layer's own inputs change. Here, `COPY pyproject.toml
uv.lock ./` followed by `RUN uv sync ...` means that expensive
dependency-install layer's cache key is tied **only** to those two files.

If the whole `src/` folder were copied in alongside them before installing,
any edit to `serve.py` would invalidate that `COPY`, and — because Docker's
cache is strictly sequential — every layer after it, including the
multi-minute dependency install, would be forced to re-run too, even though
the dependency set never changed.

This was directly confirmed in this project: after editing `serve.py` to fix
the CUDA-tensor-on-CPU loading bug, the rebuild log showed every earlier
layer (`uv sync`, the CPU-torch `uv pip install`, etc.) as `CACHED`, and only
`COPY src/ ./src/` plus the final image export actually ran — the whole
rebuild finished in under a second instead of the ~40 minutes the dependency
install had taken the first time.

## Q5 — Single-stage vs. multi-stage image size

**Not measured yet in this project** — this needs an actual naive
single-stage build plus `docker history food11-api:latest` run on your
machine to report real numbers; I don't have that output to give you
honestly. What's true by construction, without needing to measure it: the
multi-stage build here entirely discards the `builder` stage — which
contains `pip`, the `uv` installer itself, and the dependency-resolution
process — from the final image. The runtime stage starts fresh from
`python:3.14-slim` and only receives the already-built `.venv` plus `src/`,
so at minimum it avoids shipping two copies of the interpreter/build tooling
that a single-stage build would carry. Run `docker history
food11-api:latest` to see the actual largest layers (expect the CPU
`torch`/`torchvision` install and the base image itself to dominate).

## Q6 — What breaks/slows down without `.dockerignore`

Without it, every `docker build` would first transfer `.venv/` (a large,
host-OS-specific virtualenv you'd never want to ship — the container builds
its own), `data/` (over a gigabyte of images, per Lab 1), `mlruns/`/
`mlflow.db` (local run history), and `.git/` (the entire repo history) into
the Docker build context, before the build even starts — dramatically
slowing the "transferring context" step we saw at the top of every build log
tonight, even though none of it is ever actually `COPY`-ed into the image
(only `pyproject.toml`, `uv.lock`, and `src/` are).

The one most likely to actually **break** the build, not just slow it:
`.venv/` — a Windows-built virtualenv contains OS-specific binaries and
symlinks that have no business being copied into a Linux build context, and
could confuse or crash tooling that scans the full context. `data/` at over
a gigabyte risks hitting practical context-size/time limits well before
anything semantically breaks.

## Q7 — Why not `127.0.0.1:5000`; what `host.docker.internal` resolves to

Every container gets its own isolated network namespace by default —
`127.0.0.1` *inside* the container refers to the container itself, not the
Windows host. A request to `127.0.0.1:5000` from inside the container looks
for a server running inside that same container, which doesn't exist (the
mlflow server runs directly on the host, in a separate PowerShell process).

`host.docker.internal` is a special DNS name Docker Desktop provides
specifically to solve this — it resolves to the host machine's own address
as seen from inside the container's network namespace, letting a
containerized process reach services running directly on Windows.

**Real, hard-won addendum from this project**: reaching the mlflow server
via `host.docker.internal` wasn't automatically enough. Newer mlflow
versions validate the incoming HTTP `Host` header to block DNS-rebinding
attacks, and by default only allow `localhost`/`127.0.0.1`-style hosts. A
request arriving with `Host: host.docker.internal:5000` got rejected with a
403 (`Invalid Host header - possible DNS rebinding attack detected`) until
the mlflow server was restarted with:
```
MLFLOW_SERVER_ALLOWED_HOSTS=localhost,localhost:*,127.0.0.1,127.0.0.1:*,host.docker.internal,host.docker.internal:*
```
Two details that mattered getting this right: setting this env var
*replaces* mlflow's default allowlist rather than extending it (so
`localhost`/`127.0.0.1` had to be re-listed explicitly), and the match
against each allowed entry is an exact string comparison unless the entry
itself contains a `*` — the actual `Host` header always includes the port
(`host.docker.internal:5000`), so entries without a `:*` wildcard suffix
never match.

## Q8 — Restarting a container from the same image without rebuilding

**Not run yet in this session** — worth actually doing: `docker stop
<container_id>`, then `docker run ...` again from the same
`food11-api:latest` image (same mount), and confirm `/predict` still works
with no `docker build` in between.

What it demonstrates either way: this `Dockerfile` never copies model
weights into the image — `load_model()` fetches them at container
**startup**, over the network from the mlflow server, reading the actual
bytes from the mounted `mlruns` folder. So the image only bakes in code +
Python dependencies; the model itself is fetched fresh every time a
container starts. A corollary worth confirming alongside this test: if the
mlflow server isn't running, or the `mlruns` mount is missing, a fresh
container from this exact same image would fail at startup — proving the
image alone isn't self-contained for serving, by design.

## Q9 — What's missing before another machine could pull and run this image

Two separate gaps, not one:

1. **The image itself has never been pushed anywhere.** It only exists in
   this one machine's local Docker Desktop image cache — `docker tag` +
   `docker push` to a registry (Docker Hub, GHCR, ECR, etc.) is the missing
   step before `docker pull food11-api:latest` would work from a CI runner
   or a Kubernetes cluster.

2. **Even with the image pushed, this exact setup wouldn't run correctly
   elsewhere**, because the container depends on two things that only exist
   on this one Windows laptop and were never made portable: the mlflow
   tracking server itself (`uv run mlflow server ...`, reachable only via
   `host.docker.internal` from a Docker Desktop host — a hostname with no
   meaning on a Linux CI runner or inside a k8s pod), and the `mlruns`
   artifact folder, mounted in from a literal local path and recorded
   internally as a `file:C:/Users/...` URI (see Lab 2 Q6). Neither the
   tracking server nor the artifact store is network-reachable or portable
   outside this laptop. A CI runner or k8s cluster would need a properly
   hosted, network-accessible mlflow server with a real artifact store
   (S3/Azure Blob-backed, not a local folder) before this image could serve
   correctly anywhere else.
