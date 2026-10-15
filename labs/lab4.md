# Lab 4 — Orchestrating the App with Docker Compose

Answers are grounded in the files actually committed on
`orchestrate-docker-compose` (`docker-compose.yml`, `mlflow/Dockerfile`,
`frontend/Dockerfile`, `frontend/app.py`) and in what happened while getting the
stack running. Where a question describes an experiment that wasn't run as a
deliberate exercise, the answer says whether it was observed incidentally or
only reasoned from how Docker/mlflow behave. Fill in any "not run" part with
your own observation if you run it.

## Deviations from the lab text (and why)

- **`Staging` → the `champion` alias.** The lab says "move it to Staging".
  Registry stages are deprecated in current mlflow, and this project has used
  the `champion` alias since Lab 3 (`models:/food11@champion`). Wherever the lab
  says "Staging", read "the `champion` alias".
- **mlflow serves artifacts over HTTP.** The lab's `mlflow/Dockerfile` uses
  `--default-artifact-root /mlflow-data/mlruns`. With a plain local path, every
  client reads and writes model files straight at that path on *its own*
  filesystem — the same path-coupling that broke Lab 3 (`file:C:/Users/...`
  could not be resolved inside a Linux container). The committed version uses
  `--serve-artifacts --artifacts-destination /mlflow-data/mlruns
  --default-artifact-root mlflow-artifacts:/`, so artifacts go through the
  server's HTTP API while the bytes still land on the `mlflow-data` volume.
  Confirmed: a new experiment's `artifact_location` prints `mlflow-artifacts:/1`,
  and the `inference` container logs `Downloading artifacts: 100% 8/8` at
  startup with no file mounts at all.
- **Host header allowlist.** `mlflow` sets `MLFLOW_SERVER_ALLOWED_HOSTS` to
  include `mlflow` and `mlflow:*` (see Q3).
- **Network-hardened Dockerfiles.** `mlflow/Dockerfile` and
  `frontend/Dockerfile` use `pip --timeout 120 --retries 5` and a BuildKit pip
  cache mount (`RUN --mount=type=cache,...`), because plain `pip install`
  repeatedly timed out on a slow connection and every retry started from zero.

## Q1 — `/mlflow-data` with no volume, run standalone

Everything written there lives in the container's **writable layer**, which
exists only as long as that container does. Stopping the container keeps it
(`docker start` would bring the data back), but `docker rm` deletes the layer
and the data with it. A new container started from the same image begins from
the image's contents, which have no `mlflow.db` or `mlruns`, so the UI would
show an empty `Default` experiment and no registered models.

*Not run as a standalone experiment.* This follows from how image layers and
container layers work; the `down -v` observation in Q10 shows the same
"empty again" result for the volume case.

## Q2 — Named volume vs. bind mount

A bind mount to a folder in the repo *would* work functionally, but a named
volume is the better fit here:

- **It's Docker-managed storage on the Linux side.** On Windows with
  Docker Desktop/WSL2, a bind mount of a Windows folder crosses the
  Windows↔WSL file-sharing boundary, which is slower and has a history of
  file-locking quirks — a poor match for a SQLite file that mlflow writes
  constantly. A named volume lives on the VM's own Linux filesystem.
- **It's portable.** The compose file doesn't bake in a host path like
  `C:\Users\96171\...`, so the same file works on any machine. Host-path
  coupling is exactly what made the Lab 3 artifact URIs unusable elsewhere.
- **It keeps run output out of the repo.** A bind mount into the project folder
  would put `mlflow.db`/`mlruns` back in the working tree, where they would need
  gitignoring again (Lab 2 Q3 — and that `.gitignore` was already silently
  broken once by a UTF-16 encoding bug).

A bind mount is still the right tool for local dev when you want to *edit* files
on the host and see them inside the container; run-time state like a database
isn't that case.

## Q3 — Why `http://mlflow:5000` resolves now

`docker compose` creates a user-defined network for the project and attaches
every service to it. On user-defined networks Docker runs an embedded DNS
server that resolves each **service name** to that container's current IP, so
`mlflow` becomes a valid hostname for the other containers. The plain
`docker run` containers in Lab 3 sat on the default bridge network, which has
no name resolution, and the mlflow server wasn't a container at all — it ran
on the Windows host, reachable only through `host.docker.internal` (or
`--network host` on Linux).

One extra requirement discovered along the way: mlflow validates the HTTP
`Host` header to block DNS-rebinding attacks. From the installed source
(`mlflow/server/security_utils.py`), the default allowlist is localhost
variants plus private IP ranges, and the match is an exact string comparison
unless the entry contains a `*`. A request to `http://mlflow:5000` carries
`Host: mlflow:5000`, which isn't in that default list, so the compose file sets
`MLFLOW_SERVER_ALLOWED_HOSTS: "localhost,localhost:*,127.0.0.1,127.0.0.1:*,mlflow,mlflow:*"`.
Setting the variable *replaces* the defaults rather than extending them, which
is why `localhost` and `127.0.0.1` are re-listed. (This was set up front from
the source; the same 403 was seen first-hand in Lab 3 with
`host.docker.internal`.)

## Q4 — Why `INFERENCE_URL` comes from an environment variable

A hardcoded `http://inference:8000` only resolves *inside* the Compose network.
If the frontend image were run alone with `docker run`, or `app.py` were run
directly on the host with `streamlit run`, that name wouldn't resolve and every
upload would fail. Reading `os.environ.get("INFERENCE_URL", "http://127.0.0.1:8000")`
keeps one image usable everywhere: Compose overrides the value to
`http://inference:8000`, local development uses the `127.0.0.1` default, and any
other deployment can point it somewhere else — without rebuilding the image.

## Q5 — Why `inference` publishes no port

Only things a human opens in a browser need a host port: the mlflow UI (5000)
and the frontend (8501). `inference` is only ever called by the frontend, over
the Compose network via its service name (`http://inference:8000`), and that
traffic never touches the host's ports. Leaving it unpublished also shrinks the
attack surface — the prediction API isn't reachable from the host or the local
network at all. The observable consequence: you can't `curl` the inference API
from the host; you test it through the frontend (or `docker compose exec`).

## Q6 — `depends_on` is start order, not readiness

`depends_on` only waits for the `mlflow` *container process* to start, not for
the tracking server inside it to be accepting requests (a fresh database also
runs migrations first). `serve.py` calls `mlflow.pyfunc.load_model(...)` inside
its startup handler. If mlflow isn't answering yet, that call raises, FastAPI's
startup fails (`Application startup failed. Exiting.`), uvicorn exits, and the
`inference` container exits — with no restart policy set, it stays down until
someone starts it again. `docker compose logs inference` shows the traceback.

*Not deliberately reproduced.* The same failure mode — an exception in startup
ending in `Application startup failed. Exiting.` — was seen several times in
Lab 3 (Host header 403, artifact path error, CUDA deserialization error).
Typical fixes: a `healthcheck` on `mlflow` with `depends_on: condition:
service_healthy`, a `restart: on-failure` policy on `inference`, or a retry loop
around the model load in `serve.py`.

## Q7 — `docker compose ps`

Actual output (trimmed):

```
mlops-lab-1-frontend-1    ...  0.0.0.0:8501->8501/tcp, [::]:8501->8501/tcp
mlops-lab-1-inference-1   ...  8000/tcp
mlops-lab-1-mlflow-1      ...  0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp
```

`mlflow` and `frontend` have published ports (container port mapped to the
host); `inference` shows only `8000/tcp`, meaning the port is exposed inside
the network but not mapped to the host. That matches `docker-compose.yml`, which
has a `ports:` entry only for those two services (Q5).

## Q8 — New Staging/`champion` version: which model answers?

The **old** version keeps answering. `serve.py` loads the model once, at
startup, into the running process's memory; repointing the alias in the
registry changes nothing for a process that has already loaded its model. The
command that picks up the new version without rebuilding anything:

```powershell
docker compose restart inference
```

*The register-a-new-version-and-compare run wasn't done for this write-up.* To
see it concretely: train a second run with different settings, register it as
version 2, run
`MlflowClient().set_registered_model_alias('food11', 'champion', 2)`, upload the
same image before and after the restart, and note whether the prediction
changes (it can, since the two runs have different weights — the version 1 model
from earlier already mislabels `Bread/0_0.jpg` differently than the Lab 3 model
did).

## Q9 — Why `restart` alone is enough

Restarting the container re-runs the startup handler, which calls
`load_model("models:/food11@champion")` again; the alias is resolved afresh and
now points at the new version. No rebuild is needed because the **image only
bakes in code and dependencies** — `serve.py` and the Python environment. The
model weights are **not** in the image; they're fetched from mlflow every time
the container starts. This is visible in the logs: each startup prints
`Downloading artifacts: 100%|██████████| 8/8` before `Application startup
complete`, with no model file ever copied into the image.

## Q10 — `down`/`up` vs. `down -v`/`up`

- **`docker compose down` then `up`**: the registered model and the `champion`
  alias are still there. `down` removes the containers and the network but
  leaves named volumes alone, and the data lives on the `mlflow-data` volume.
  Observed incidentally: after the mlflow container was recreated with
  `docker compose up -d --build mlflow`, the `food11` experiment created
  earlier was still present (no "does not exist, creating" message), so
  volume-backed data survives container replacement.
- **`docker compose down -v` then `up`**: `-v` also deletes the named volumes,
  so mlflow starts with a brand-new empty database. Observed incidentally:
  after `down -v`, the next `set_experiment('food11')` logged
  `Experiment with name 'food11' does not exist. Creating a new experiment.`,
  and the model had to be retrained, re-registered (`food11` was created from
  scratch) and re-aliased before `inference` could start. The difference is
  entirely the volume: containers are disposable, the volume is the state.

*These were seen while troubleshooting rather than as a planned exercise.*

## Q11 — What Compose can't give you here

**Three replicas behind a load balancer:** Compose is a single-host tool. It
can start several copies of a service, but it provides no real load balancer in
front of them, no automatic scaling or rolling updates, and no placement of
replicas across machines. Publishing a fixed port can't be shared by multiple
copies on one host. That needs an orchestrator — Kubernetes, Docker Swarm, ECS
and similar — with a Service/Ingress or cloud load balancer, health-based
routing, and rescheduling of failed replicas.

**mlflow surviving a machine failure:** here mlflow is a single container with a
SQLite file on a volume on one machine, which makes it a single point of
failure. If the machine dies, the tracking server, the registry and the
artifacts go with it. Surviving that needs the state moved off the host: a
managed database such as PostgreSQL for the backend store, a remote object
store such as S3/GCS/Azure Blob for artifacts (`--artifacts-destination`),
backups, and more than one mlflow instance behind a load balancer. Also still
missing from earlier labs: the built images only exist in this machine's local
cache and have never been pushed to a registry (Lab 3 Q9), and secrets,
monitoring and centralized logging aren't handled by this compose file.
