# Lab 1 — Git/DVC and Data Preparation

Answers below are grounded in what actually happened in this repo (commit
hashes, file diffs, and current file contents), not the generic/expected
answer.

## Q1 — Files created by `uv init`, and their purpose

`uv init` (commit `7ef372e`, "Init uv project") created:

- **`pyproject.toml`** — project metadata (name, version, description,
  authors) and the dependency list (`dvc`, later `pillow`). This is the
  single source of truth for what the project needs to run.
- **`uv.lock`** — the fully resolved, pinned dependency tree (exact
  versions/hashes for every package, direct and transitive) so `uv sync`
  reproduces an identical environment on any machine.
- **`.python-version`** — pins the interpreter version (`3.14`) that `uv`
  should provision/use for this project.
- **`README.md`** — empty placeholder for project documentation.
- It also appended `.idea/` to `.gitignore` (PyCharm project files).

Note: `main.py` is *not* a `uv init` artifact here — it was already created
one commit earlier, in `7e5e223` ("Initialize git and dvc"), as a leftover
PyCharm bootstrap script. `uv init` did not overwrite it.

## Q2 — Files DVC creates, what they do, and what gets committed

`dvc init` (commit `7e5e223`) created:

- **`.dvc/config`** — the DVC project config (remotes, cache settings,
  etc.). Started empty, later gained `['remote "origin"']` and
  `['remote "localstorage"']` sections.
- **`.dvc/.gitignore`** — tells *git* to ignore DVC's own local-only
  internals:
  ```
  /config.local
  /tmp
  /cache
  ```
- **`.dvcignore`** — the DVC analogue of `.gitignore`: patterns DVC itself
  should skip when scanning the working directory (empty/default here).

Later, `dvc add data` (commit `18cf8d2`) created:

- **`data.dvc`** — the pointer/metadata file standing in for the actual
  data directory (see Q5).

What should be committed to GitHub: `.dvc/config`, `.dvc/.gitignore`,
`.dvcignore`, and `data.dvc` — all of these were committed in this repo,
and all of them are small, human-readable, and contain no data or secrets.
`.dvc/config.local` (see Q3) and `.dvc/cache` must never be committed;
`.dvc/.gitignore` already enforces that automatically.

## Q3 — Credential storage and version control

DVC stores remote credentials outside `.dvc/config` specifically so they
never get committed. The mechanism is `dvc remote modify <remote> --local
<key> <value>`, which writes to **`.dvc/config.local`** instead of
`.dvc/config`. `.dvc/.gitignore` ignores `/config.local` by default, so a
locally-configured credential is never picked up by `git add`. That file
does not exist in this repository's history — there's no commit touching
it, which is exactly the expected/correct outcome.

`--local` is one of three scopes `dvc remote modify` supports besides the
implicit "project, committed" default:

- `--local` — this repo's `.dvc/config.local` (gitignored, machine + repo
  specific). Used for the DagsHub token in this project.
- `--global` — `~/.config/dvc/config` (or platform equivalent), applies to
  every DVC project for that user.
- `--system` — a machine-wide config shared by all users.

Credentials should **not** be version-controlled under any of these
options — they're secrets tied to a person/machine, not to the project's
logic, and committing them would leak access to whoever can read the git
history (forever, even if later removed).

## Q4 — `.gitignore` change after `dvc add data`

Before `dvc add data`, `.gitignore` only had:

```
.idea/
```

`dvc add data` (commit `18cf8d2`) appended one line:

```diff
 .idea/
+/data
```

`dvc add` automatically adds the path it just tracked to `.gitignore` so
git never accidentally picks up the real data files. This is the
mechanical link between the two tools: DVC takes ownership of everything
under `data/` (via `data.dvc` + its cache), and tells git to stay out of
that directory entirely — git only ever sees the one small pointer file.

## Q5 — What's in `data.dvc` and what it does

Current contents:

```yaml
outs:
- md5: 36598702f6335bfb2b0b9510b7d1dfcb.dir
  size: 1277237512
  nfiles: 36578
  hash: md5
  path: data
```

- **`path: data`** — the tracked directory, relative to this file.
- **`md5: ....dir`** — not a hash of file *contents* directly; it's the md5
  of a manifest DVC builds internally that lists every file in `data/`
  along with each file's own md5. The `.dir` suffix marks it as a
  directory-level entry. This single hash is what changes any time a file
  is added, removed, or modified anywhere under `data/`.
- **`size`** — total size in bytes of everything under `data/` (~1.19 GB).
- **`nfiles`** — total file count (36,578).

Function: `data.dvc` is the pointer git actually tracks in place of the
real data. It lets DVC reconstruct the exact state of `data/` for any git
commit — `dvc checkout` reads whichever version of `data.dvc` is currently
checked out and uses its hash to fetch/link the matching files from the
DVC cache/remote into the working directory. It is the mechanism behind
the checkout test in Q8.

The file's own history shows this hash changing exactly when the data
changed:

| Commit | nfiles | What changed |
|---|---|---|
| `18cf8d2` "Add Food-11 raw dataset" | 16,643 | raw images only |
| `a6ff549` "Add food11_processed and food11_processed_mini" | 36,578 | + processed + mini processed outputs |

## Q6 — Presence on GitHub vs. DagsHub, and the pointer files

**GitHub** (`git push`): code and pointer files are present —
`src/food11/data.py`, `pyproject.toml`, `uv.lock`, `main.py`, `README.md`,
and the DVC pointer/metadata files `data.dvc`, `.dvc/config`,
`.dvc/.gitignore`, `.dvcignore`. The actual `data/` directory is **not**
on GitHub — it's gitignored (Q4), so git never had it to push in the first
place.

**DagsHub**: this is where it didn't go as expected. DagsHub was
originally configured as the DVC remote (`dvc remote add origin
https://dagshub.com/theomwd711/mlops-lab-1.dvc`, commit `708e079`, set as
default in `d533979`). `dvc push` to it repeatedly failed — server
disconnects/failed transfers — because of the sheer number of small files
(16,643 raw images alone, 36,578 total). This turned out to be a
class-wide problem; the professor's official follow-up offered two
workarounds, and I took **Option 1**: switch the default DVC remote to a
local folder outside the repo instead of DagsHub (commit `2672db0`,
`.dvc/config` now points `core.remote` at `localstorage`, a path on the
local machine). `dvc push` to that local remote succeeded for all 36,578
files.

Net effect: the pointer files (`data.dvc`, `.dvc/config`) are on GitHub as
intended, but the actual data content was never successfully pushed to
DagsHub for this project — it lives only in the local DVC remote/cache.
DagsHub still shows the *code* (mirrored from GitHub if connected) but not
usable data, since no push to it ever completed.

## Q7 — Cloning the repo somewhere new

A fresh `git clone` brings over everything git tracks: the code and the
pointer files (`data.dvc`, `.dvc/config`, `.dvc/.gitignore`,
`.dvcignore`). It does **not** bring the actual contents of `data/` — that
directory is gitignored and was never part of any git commit, so after
cloning, `data/` simply doesn't exist on disk (or exists only if something
else created it).

To materialize the data, the clone needs `dvc pull` (or `dvc checkout` if
the objects are already in a local cache), which reads `data.dvc`,
resolves the hash to objects in the configured DVC remote, and copies them
into `data/`.

Caveat specific to this repo: the default remote is `localstorage`, a path
on the original machine's filesystem (`C:\Users\...\dvc-storage\mlops-lab-1`,
outside the repo). `dvc pull` will only work on a clone that can actually
reach that path — i.e. the same machine, or one with that folder mounted
or otherwise made available. This is the real cost of the Option 1
workaround: it made `dvc push` reliable, but it also means the remote is
no longer something an arbitrary collaborator's clone can pull from the
way a hosted remote (DagsHub, S3, etc.) would allow. A teammate cloning
this repo elsewhere would need the `localstorage` path (or a re-pointed
remote) to actually get the data.

## Q8 — Checking out an earlier commit and `dvc checkout`

Test performed: checked out `0133698` ("Add data preparation script") —
the last commit before `food11_processed`/`food11_processed_mini` existed,
where `data.dvc` still pointed at the raw-only hash (16,643 files) — then
ran `dvc checkout`. Result: `food11_processed/` and
`food11_processed_mini/` disappeared from `data/`, leaving only
`food11_raw/`. Checking out `main` again and re-running `dvc checkout`
brought both processed folders back.

Why this happens: `git checkout` only moves which *version* of `data.dvc`
is active — it's a tiny text file, so that part is instant and doesn't
touch anything under `data/`. Git has no idea what's inside the actual
data directory; it never tracked it. `dvc checkout` is the separate step
that reads whatever hash is currently in `data.dvc` and makes the real
`data/` directory match it, by linking/copying the corresponding objects
from the DVC cache and removing anything that isn't part of that hash's
manifest. At commit `0133698`, the tracked hash corresponds to a `data/`
that only contains `food11_raw/` — so `dvc checkout` correctly deleted the
processed folders, since (from that commit's point of view) they were
never supposed to exist. This is the core git/DVC split in practice: git
versions the *pointer*, DVC materializes the *content* that pointer
refers to, and the two only stay in sync when you run both `git checkout`
and `dvc checkout` together.
