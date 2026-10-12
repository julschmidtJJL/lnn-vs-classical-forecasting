# Cloud test run — bare infra smoke

Goal: prove the pipeline runs end-to-end on a fresh Linux box (install →
data fetch → all models → DM tests → Supabase logging), **not** produce
publishable results. Uses `src/config.smoke.yaml`: 1 ticker (AAPL), ~3
walk-forward folds. Full grid is `config.yaml` (3 tickers, 10y history, 5 seeds).

---

## Pre-run setup (ONE-TIME, before any run) — provenance + scaling

The scale+repro work added requirements that a run now depends on:

1. **Apply DB migrations** in the Supabase SQL editor (idempotent, safe to re-run):
   `src/loggers/migrations/002_experiment_results_provenance.sql` and
   `003_experiment_summaries_provenance.sql`. Without them the new
   `run_id/condition/seed` columns + upsert indexes don't exist and logging fails.
2. **Create the Storage bucket** `experiment-snapshots` in Supabase and give the
   anon role insert+select on it (mirrors the table RLS posture — the anon key is
   treated as semi-public for this private project). This is where the input
   parquet snapshots + per-fold predictions land.
3. **`GIT_COMMIT` is now ENFORCED.** The image has no `.git`, so a run aborts
   unless you pass the commit. Always launch with
   `-e GIT_COMMIT=$(git rev-parse HEAD)` (shown below). For a throwaway local run
   you may instead set `-e ALLOW_NO_GIT_COMMIT=1`.
4. `pyarrow` (parquet engine) is in `requirements.txt` — no action, just noted.

---

## 0. Local dry-run (do this first)

### a) Code path, no Docker (fastest feedback)
From the repo root, with your venv active:
```bash
cd src && python main.py config.smoke.yaml
```
Expect: `[WF] 3 folds (initial_window=1180, horizon=21)`, all model runs, a
DM/Holm summary, and new rows in the `experiment_summaries` Supabase table.

### b) Docker image build + run (the real cloud dress-rehearsal)
Requires Docker Desktop. The Dockerfile pins `--platform=linux/amd64` to match
the x86_64 VM; on an Intel Mac that's native, on Apple Silicon it builds via
emulation (slower but fine):
```bash
cp .env.example .env   # then fill in real values (gitignored, never committed)
docker build -t thesis-exp .
docker run --rm -e GIT_COMMIT=$(git rev-parse HEAD) --env-file .env thesis-exp   # config.smoke.yaml by default
```
Running the experiment *inside* the container (not just the venv) is what
validates the real cloud path: it proves the Linux wheels install AND that the
container can reach Yahoo/FRED/SEC and write to Supabase. If this passes
locally, Hetzner is the same image on someone else's box.

---

## 1. Provision a Hetzner VM (~5 min)

1. Create an account at <https://console.hetzner.cloud>.
2. New Project → **Add Server**:
   - Location: a EU region (Nuremberg/Falkenstein).
   - Image: **Ubuntu 24.04**.
   - Type: **CPX31** (4 vCPU / 8 GB) — ~€0.02/hr.
   - SSH key: add yours (or set a root password).
3. Create. Note the public IP.

## 2. Set up the box (~5 min)
```bash
ssh root@<IP>

# Docker
curl -fsSL https://get.docker.com | sh

# Get the code. The repo is PRIVATE (hardcoded keys), so cloning needs auth:
apt-get update && apt-get install -y git
#   git clone https://<GITHUB_PAT>@github.com/julschmidtJJL/BachelorThesisExperiment.git thesis
# (use a GitHub Personal Access Token) — OR copy from your Mac, no token needed:
#   scp -r ./BachelorThesisExperiment root@<IP>:~/thesis   # venv is large; exclude it if you can
cd thesis
```

## 3. Create .env, build + run the smoke test
`.env` is gitignored / not in the image, so create it on the box (paste your
real values, or `scp` your local `.env` over):
```bash
cp .env.example .env
nano .env                      # fill in SUPABASE_*, FRED_API_KEY, etc.

docker build -t thesis-exp .
docker run --rm -e GIT_COMMIT=$(git rev-parse HEAD) --env-file .env thesis-exp
```
Watch the logs scroll. A clean finish ends with the per-ticker summary and no
traceback. (Missing a required var fails fast with a clear message.)

## 4. Confirm success
- Supabase: new rows in `experiment_summaries` for this `run_id`
  (printed at the top of the run as `run_id=...`).
- The run printed `[WF] 3 folds` and a DM significance table.

## 5. Tear down (stop paying)
```bash
exit                       # leave the VM
```
In the Hetzner console: **Delete** the server. (Powered-off servers still bill
for disk — delete it.)

---

## Going from smoke → full grid
Once infra is proven, run the real config on a longer-lived box:
```bash
docker run --rm -e GIT_COMMIT=$(git rev-parse HEAD) --env-file .env thesis-exp python main.py config.yaml
```
**Runtime note:** the full grid is now 3 tickers × **5 seeds** × **10y history**
(~80 folds) × **50 epochs**, so it is much longer than the old ~6–8h/1-seed run —
budget accordingly and size the box up (cost is trivial). Crash-resume is keyed
on `(run_id, ticker)` per seed (`run_id = {RUN_ID}#s{seed}`); re-launch with the
same `RUN_ID` to resume — completed (ticker, seed) pairs are skipped, and seed 1's
input snapshot is reused by later seeds. To run a quick single-seed pass, set
`experiment.seeds: [42]` in the config.

## If the Docker build fails on a wheel
Some scientific packages occasionally need a compiler if no wheel matches.
Add this above the `pip install` line in the `Dockerfile`:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
```
