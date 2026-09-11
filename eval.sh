#!/usr/bin/env bash
# Evaluate every submission listed in weekX.csv.
#
# Per row:
#   1. sparse-clone only the GitHub submission folder (/tree/<branch>/<subdir>)
#   2. download the compressed model from HuggingFace
#   3. zero visual tensors in-place (ensure_visual_zero.py)
#   4. create/reuse a local .venv from pyproject.toml
#   5. decompress with the student's decompress.py
#   6. launch vLLM on a free port and wait until ready
#   7. run evaluation/run_eval.py and append metrics to WeekX.results.csv
#
# Resumable: re-running skips completed (status=ok) rolls, and for an in-progress
# roll reuses an existing clone / download / decompress / live vLLM server when
# the server is healthy and its --model path matches the decompressed checkpoint.
# On failure, artifacts + vLLM are left in place for the next run.
#
# Usage:
#   GITHUB_USERNAME=... GITHUB_TOKEN=... ./eval.sh --week <N> [weekX.csv] [WeekNN.results.csv] [--interact] [--slack_log]
#
# Modes:
#   (default)   Process every CSV row; Slack on success/failure; continue after failures.
#               Failed-run artifacts are kept; vLLM is wound down before the next row.
#   --interact  Stop at the first failure; keep that run's artifacts + vLLM for debugging.
#               Also accepted as env INTERACT=1.
#   --slack_log Append Err/err greps from eval + vLLM logs to the Slack message.
#               Also accepted as env SLACK_LOG=1.
#
# Required:
#   --week N | WEEK=N     Course week number (e.g. 4 or 04 → Week04). Used to validate paths.
#   GITHUB_USERNAME       GitHub username for private repo clones
#   GITHUB_TOKEN          GitHub PAT / token for private repo clones
#   SLACK_WEBHOOK_URL     Slack incoming webhook (used by slack.py)
#
# Naming (validated before any eval work), matching the course tree:
#   CS6013/<roll_number>/Week<WW>/Compression_<target>/Submission<NN>/
#     compress.py, decompress.py, pyproject.toml, README.md
#     compression/__init__.py
#     decompression/__init__.py
#   HuggingFace: <roll>-Week<WW>-Compression<target>-Submission<NN>
#                (hyphen after Compression is optional: Compression40 or Compression-40)
#
# Optional env overrides:
#   WORK_ROOT LIMIT MAX_CONCURRENCY CONFIG DATASET VLLM_READY_TIMEOUT_SEC INTERACT SLACK_LOG
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INTERACT="${INTERACT:-0}"
SLACK_LOG="${SLACK_LOG:-0}"
WEEK="${WEEK:-}"
POSITIONAL=()
args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --interact|interact)
      INTERACT=1
      ;;
    --slack_log|slack_log)
      SLACK_LOG=1
      ;;
    --week|-w)
      i=$((i + 1))
      if [[ $i -ge ${#args[@]} ]]; then
        echo "ERROR: ${arg} requires a week number (e.g. --week 4)." >&2
        exit 1
      fi
      WEEK="${args[$i]}"
      ;;
    --week=*|-w=*)
      WEEK="${arg#*=}"
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
  i=$((i + 1))
done

if [[ -z "$WEEK" ]]; then
  echo "ERROR: Week is required. Pass --week <N> or set WEEK=<N> (e.g. --week 4)." >&2
  exit 1
fi

# Normalize to WeekNN (zero-padded 2 digits). Accepts 4, 04, Week4, Week04.
WEEK_NUM="$(python3 -c 'import re,sys; s=sys.argv[1].strip(); m=re.search(r"(\d+)", s); print(m.group(1) if m else "")' "$WEEK")"
if [[ -z "$WEEK_NUM" ]]; then
  echo "ERROR: Could not parse week number from '${WEEK}'." >&2
  exit 1
fi
WEEK_LABEL="$(printf 'Week%02d' "$((10#$WEEK_NUM))")"

CSV_IN="${POSITIONAL[0]:-weekX.csv}"
RESULTS_CSV="${POSITIONAL[1]:-${WEEK_LABEL}.results.csv}"
WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}/eval_workdir}"
CONFIG="${CONFIG:-configs/eval_config.yaml}"
# Prefer env DATASET / LIMIT when set; otherwise the local simple-math sample set.
DATASET="${DATASET:-datasets/CS6013_sample_math_format_dataset}"
LIMIT="${LIMIT:-30}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-30}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32000}"
# Prompt budget (~1k) + generation budget → vLLM KV-cache / memory sizing.
MAX_MODEL_LEN="$((1000 + MAX_NEW_TOKENS))"
MODEL_NAME_DECOMPRESS="${MODEL_NAME_DECOMPRESS:-Qwen-3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-3.5-4b}"
VLLM_READY_TIMEOUT_SEC="${VLLM_READY_TIMEOUT_SEC:-1800}"
# Qwen3.5 GDN/Mamba: CUDA graphs need max_num_seqs <= Mamba cache blocks.
# Low --gpu-memory-utilization shrinks those blocks; keep seqs modest.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
# Original Qwen3.5-4B text-tower size (GiB) for size_frac = compressed_text / original_text.
ORIGINAL_TEXT_GB="${ORIGINAL_TEXT_GB:-8.0585}"

echo "[config] WEEK=${WEEK_LABEL}  DATASET=${DATASET}  LIMIT=${LIMIT}  MAX_CONCURRENCY=${MAX_CONCURRENCY}  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}  MAX_MODEL_LEN=${MAX_MODEL_LEN}  VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}  VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"

GITHUB_USERNAME="${GITHUB_USERNAME:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [[ -z "$GITHUB_USERNAME" || -z "$GITHUB_TOKEN" ]]; then
  echo "ERROR: Set GITHUB_USERNAME and GITHUB_TOKEN in the environment before running." >&2
  echo "  Example: GITHUB_USERNAME=me GITHUB_TOKEN=ghp_... ./eval.sh" >&2
  exit 1
fi

if [[ "$INTERACT" == "1" || "$INTERACT" == "true" || "$INTERACT" == "yes" ]]; then
  INTERACT=1
  echo "[mode] interact=ON — stop on first failure; keep failed-run artifacts"
else
  INTERACT=0
  echo "[mode] interact=OFF — process all rows; Slack on each result"
fi

if [[ "$SLACK_LOG" == "1" || "$SLACK_LOG" == "true" || "$SLACK_LOG" == "yes" ]]; then
  SLACK_LOG=1
  echo "[mode] slack_log=ON — include error greps in Slack"
else
  SLACK_LOG=0
fi

mkdir -p "$WORK_ROOT" logs outputs

echo "[setup] Syncing .venv from pyproject.toml ..."
uv sync --extra cuda129
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/.venv/bin/activate"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
HF_CLI="${SCRIPT_DIR}/.venv/bin/hf"

if [[ ! -f "$RESULTS_CSV" ]]; then
  printf '%s\n' \
    "roll,name,email,compression_target,github_url,huggingface_url,accuracy,parse_rate,num_correct,num_examples,size_frac,status,error,output_json" \
    > "$RESULTS_CSV"
elif ! head -n 1 "$RESULTS_CSV" | grep -q 'size_frac'; then
  echo "WARNING: ${RESULTS_CSV} is missing size_frac column; move it aside or delete to recreate the header." >&2
fi

find_free_port() {
  "$PYTHON" - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

dir_nonempty() {
  [[ -d "$1" ]] && find "$1" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

# Heuristic: HF-style checkpoint present under dir.
has_model_checkpoint() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  [[ -f "${d}/config.json" ]] && return 0
  [[ -f "${d}/model.safetensors.index.json" ]] && return 0
  find "$d" -maxdepth 2 \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) \
    -print -quit 2>/dev/null | grep -q .
}

# Prints three lines: clone_url, branch, subdir
# Subdir is required: we never clone the full CS6013 tree.
parse_github() {
  "$PYTHON" - "$1" <<'PY'
import re, sys
url = sys.argv[1].rstrip("/")
m = re.match(
    r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/tree/([^/]+)/(.+)$",
    url,
)
if not m:
    raise SystemExit(
        f"GitHub URL must include /tree/<branch>/<EnrollmentNo>/Week../Compression_../Submission..: {url}"
    )
owner, repo, branch, subdir = m.group(1), m.group(2), m.group(3), m.group(4)
repo = repo.removesuffix(".git")
subdir = (subdir or "").strip("/")
if not subdir:
    raise SystemExit(f"GitHub URL is missing the submission folder path: {url}")
print(f"https://github.com/{owner}/{repo}.git")
print(branch)
print(subdir)
PY
}

parse_hf_repo_id() {
  "$PYTHON" - "$1" <<'PY'
import re, sys
url = sys.argv[1].rstrip("/")
m = re.match(r"https?://huggingface\.co/([^/]+)/([^/]+)", url)
if not m:
    raise SystemExit(f"Unrecognized HuggingFace URL: {url}")
print(f"{m.group(1)}/{m.group(2)}")
PY
}

# Validate GitHub path + HuggingFace repo naming against WEEK_LABEL / roll / target.
# Tree (exact names, 2-digit Week/Submission):
#   CS6013/<roll_number>/WeekNN/Compression_<target>/SubmissionNN/
#     compress.py, decompress.py, pyproject.toml, README.md
#     compression/, decompression/
# Prints a human-readable report to stdout; exit 0 iff every row is valid.
validate_submission_formats() {
  local week_label="$1"
  shift
  "$PYTHON" - "$week_label" "$@" <<'PY'
import json
import re
import sys

week_label = sys.argv[1]  # e.g. Week04
week_m = re.fullmatch(r"Week(\d{2})", week_label)
if not week_m:
    print(f"INTERNAL: bad week_label {week_label!r}")
    raise SystemExit(2)
week_num = week_m.group(1)

# Host/scheme only; path names below are case-sensitive to match the course tree.
gh_re = re.compile(
    r"^https?://github\.com/([^/]+)/(CS6013)(?:\.git)?/tree/([^/]+)/(.+)$",
    re.IGNORECASE,
)
# CS6013/<roll_number>/WeekNN/Compression_<target>/SubmissionNN
path_re = re.compile(
    r"^([^/]+)/(Week\d{2})/(Compression_[^/]+)/(Submission\d{2})/?$"
)
hf_re = re.compile(
    r"^https?://huggingface\.co/([^/]+)/([^/]+)/?$",
    re.IGNORECASE,
)
# <roll>-WeekNN-Compression<target>-SubmissionNN  (optional hyphen: Compression40 or Compression-40)
hf_name_re = re.compile(
    r"^(.+)-(Week\d{2})-Compression-?(\d+)-(Submission\d{2})$"
)

def norm_target(raw: str) -> str:
    # CSV "40%" / "40" → GitHub Compression_40; HF Compression40 or Compression-40.
    m = re.search(r"(\d+)", (raw or "").strip())
    return m.group(1) if m else ""

def norm_roll(r: str) -> str:
    return (r or "").strip()

def expected_tree(roll: str, week: str, comp: str) -> str:
    r = roll or "<roll_number>"
    return (
        f"CS6013/{r}/{week}/{comp}/SubmissionNN/\n"
        "    compress.py\n"
        "    decompress.py\n"
        "    pyproject.toml\n"
        "    README.md\n"
        "    compression/\n"
        "    decompression/"
    )

errors: list[str] = []
ok = 0

for raw in sys.argv[2:]:
    row = json.loads(raw)
    roll = norm_roll(row.get("roll", ""))
    name = (row.get("name") or "").strip() or "(no name)"
    target = norm_target(row.get("compression_target", ""))
    gh = (row.get("github_url") or "").strip().rstrip("/")
    hf = (row.get("huggingface_url") or "").strip().rstrip("/")
    label = f"{name} ({roll})"
    row_errs: list[str] = []

    if not roll:
        row_errs.append("missing Roll no.")
    if not target:
        row_errs.append(f"unparseable Compression Target {row.get('compression_target')!r}")

    expect_gh_week = f"Week{week_num}"
    expect_gh_comp = f"Compression_{target}" if target else "Compression_<compression_target>"
    expect_hf_name = (
        f"{roll or '<roll_number>'}-{expect_gh_week}-Compression{target}-SubmissionNN"
        if target
        else f"{roll or '<roll_number>'}-{expect_gh_week}-Compression<target>-SubmissionNN"
    )
    expect_path = f"{roll or '<roll_number>'}/{expect_gh_week}/{expect_gh_comp}/SubmissionNN"

    gm = gh_re.match(gh)
    if not gm:
        row_errs.append(
            "GitHub URL must be "
            f"https://github.com/<user>/CS6013/tree/<branch>/{expect_path} "
            f"(got {gh!r})\n      expected tree:\n      {expected_tree(roll, expect_gh_week, expect_gh_comp)}"
        )
        subdir = None
    else:
        _owner, repo, _branch, subdir = gm.group(1), gm.group(2), gm.group(3), gm.group(4)
        if repo != "CS6013":
            row_errs.append(f"GitHub repo must be exactly CS6013 (got {repo!r})")

    pm = path_re.match(subdir.strip("/")) if subdir else None
    gh_enroll = gh_week = gh_comp = gh_sub = None
    if subdir and not pm:
        row_errs.append(
            "GitHub path must be exactly "
            f"{expect_path} "
            "(case-sensitive: WeekNN, Compression_<target>, SubmissionNN; two-digit week/submission). "
            f"got {subdir!r}"
        )
    elif pm:
        gh_enroll, gh_week, gh_comp, gh_sub = pm.group(1), pm.group(2), pm.group(3), pm.group(4)
        if roll and gh_enroll.lower() != roll.lower():
            row_errs.append(f"GitHub <roll_number> {gh_enroll!r} != CSV roll {roll!r}")
        if gh_week != expect_gh_week:
            row_errs.append(f"GitHub week folder {gh_week!r} != expected {expect_gh_week!r}")
        if target and gh_comp != expect_gh_comp:
            row_errs.append(
                f"GitHub folder {gh_comp!r} != expected {expect_gh_comp!r} "
                "(must be Compression_<compression_target>)"
            )
        if not re.fullmatch(r"Submission\d{2}", gh_sub):
            row_errs.append(f"GitHub submission folder must be SubmissionNN (two digits), got {gh_sub!r}")

    hm = hf_re.match(hf)
    hf_sub = None
    if not hm:
        row_errs.append(
            "HuggingFace URL must be "
            f"https://huggingface.co/<user>/"
            f"{expect_hf_name} "
            f"(got {hf!r})"
        )
        hf_repo = None
    else:
        hf_repo = hm.group(2)

    hm_name = hf_name_re.fullmatch(hf_repo) if hf_repo else None
    if hf_repo and not hm_name:
        row_errs.append(
            "HuggingFace repo name must be "
            f"{expect_hf_name} "
            f"(got {hf_repo!r}; Compression40 or Compression-40 are valid, not Compression_40; "
            "WeekNN and SubmissionNN are two digits, case-sensitive)"
        )
    elif hm_name:
        hf_enroll, hf_week, hf_tgt, hf_sub = (
            hm_name.group(1),
            hm_name.group(2),
            hm_name.group(3),
            hm_name.group(4),
        )
        if roll and hf_enroll.lower() != roll.lower():
            row_errs.append(f"HF <roll_number> {hf_enroll!r} != CSV roll {roll!r}")
        if hf_week != expect_gh_week:
            row_errs.append(f"HF week {hf_week!r} != expected {expect_gh_week!r}")
        if target and hf_tgt != target:
            row_errs.append(f"HF Compression target {hf_tgt!r} != expected {target!r}")

    # GitHub SubmissionNN === HuggingFace SubmissionNN (parsed from the course tree only).
    def _norm_submission(s: str | None) -> str | None:
        if not s:
            return None
        m = re.fullmatch(r"Submission(\d{2})", s)
        if not m:
            return None
        return f"Submission{m.group(1)}"

    gh_sub_n = _norm_submission(gh_sub)
    hf_sub_n = _norm_submission(hf_sub)
    if gh_sub_n and hf_sub_n and gh_sub_n != hf_sub_n:
        row_errs.append(
            f"GitHub {gh_sub_n} != HuggingFace {hf_sub_n} "
            "(submission numbers must match)"
        )
    if row_errs:
        errors.append(f"• {label}")
        for e in row_errs:
            errors.append(f"    - {e}")
    else:
        ok += 1

total = ok + sum(1 for line in errors if line.startswith("• "))
print(f"Format check ({week_label}): {ok}/{total} submissions valid")
if errors:
    print("Failures:")
    print("\n".join(errors))
    raise SystemExit(1)
print(
    "All submissions match CS6013/<roll_number>/WeekNN/Compression_<target>/SubmissionNN "
    "and HuggingFace <roll>-WeekNN-Compression<target>-SubmissionNN "
    "(Compression40 or Compression-40)."
)
PY
}

notify_slack_format_failure() {
  local week_label="$1"
  local report="$2"
  local roster="$3"
  local message
  message="🚫 Format check failed for ${week_label} — aborting eval (no submissions run)."
  if [[ -n "$roster" ]]; then
    message+=$'\n'"Students:"$'\n'"${roster}"
  fi
  message+=$'\n'"\`\`\`"$'\n'"${report}"$'\n'"\`\`\`"
  set +e
  printf '%s\n' "$message" | "$PYTHON" "${SCRIPT_DIR}/slack.py" -
  set -e
}

notify_slack_format_success() {
  local week_label="$1"
  local report="$2"
  local roster="$3"
  local n_subs="${4:-}"
  local message
  message="✅ Format check successful for ${week_label}"
  if [[ -n "$n_subs" ]]; then
    message+=" (${n_subs} submissions) — proceeding with eval."
  else
    message+=" — proceeding with eval."
  fi
  if [[ -n "$roster" ]]; then
    message+=$'\n'"Students:"$'\n'"${roster}"
  fi
  message+=$'\n'"\`\`\`"$'\n'"${report}"$'\n'"\`\`\`"
  set +e
  printf '%s\n' "$message" | "$PYTHON" "${SCRIPT_DIR}/slack.py" -
  set -e
}

# One line per submission: "Name (roll)"
format_students_roster() {
  "$PYTHON" - "$@" <<'PY'
import json
import sys

lines = []
for raw in sys.argv[1:]:
    row = json.loads(raw)
    name = (row.get("name") or "").strip() or "(no name)"
    roll = (row.get("roll") or "").strip() or "(no roll)"
    lines.append(f"• {name} ({roll})")
print("\n".join(lines))
PY
}

# Inject GITHUB_USERNAME / GITHUB_TOKEN into an https://github.com/... clone URL.
authenticated_clone_url() {
  local clone_url="$1"
  "$PYTHON" - "$clone_url" "$GITHUB_USERNAME" "$GITHUB_TOKEN" <<'PY'
import sys
from urllib.parse import quote, urlparse, urlunparse

url, user, token = sys.argv[1], sys.argv[2], sys.argv[3]
parsed = urlparse(url)
if parsed.scheme not in ("http", "https"):
    raise SystemExit(f"Unsupported clone URL scheme: {url}")
netloc = f"{quote(user, safe='')}:{quote(token, safe='')}@{parsed.hostname}"
if parsed.port:
    netloc = f"{netloc}:{parsed.port}"
print(urlunparse(parsed._replace(netloc=netloc)))
PY
}

# Sparse-clone only the submission folder into dest; echo that folder path.
# Never deletes an existing *successful* clone — callers must only invoke when dest is absent.
# On failure, removes any partial dest and returns non-zero with the git error on stderr.
clone_submission() {
  local github_url="$1"
  local dest="$2"
  local clone_url branch subdir auth_url
  local err_file rc

  if [[ -e "$dest" ]]; then
    echo "Refusing to clone over existing path: ${dest}" >&2
    return 1
  fi

  if ! mapfile -t _gh < <(parse_github "$github_url"); then
    echo "Failed to parse GitHub URL: ${github_url}" >&2
    return 1
  fi
  clone_url="${_gh[0]}"
  branch="${_gh[1]}"
  subdir="${_gh[2]}"

  if [[ -z "$subdir" ]]; then
    echo "Refusing full-repo clone; GitHub URL must include the submission folder: ${github_url}" >&2
    return 1
  fi

  if ! auth_url="$(authenticated_clone_url "$clone_url")"; then
    echo "Failed to build authenticated clone URL" >&2
    return 1
  fi

  mkdir -p "$(dirname "$dest")"
  err_file="$(mktemp)"

  cleanup_partial_clone() {
    rm -rf "$dest"
    rm -f "$err_file"
  }

  echo "Sparse-cloning ${clone_url} branch=${branch} folder=${subdir}" >&2
  # --no-checkout + cone sparse-checkout: only blobs under subdir are fetched.
  if ! GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse --no-checkout \
    --branch "$branch" "$auth_url" "$dest" >"$err_file" 2>&1; then
    rc=$?
    echo "git clone failed (exit ${rc}) for ${clone_url} (branch=${branch}): $(tr '\n' ' ' <"$err_file" | tail -c 1500)" >&2
    cleanup_partial_clone
    return 1
  fi
  if ! git -C "$dest" sparse-checkout set --cone "$subdir" >"$err_file" 2>&1; then
    rc=$?
    echo "git sparse-checkout set '${subdir}' failed (exit ${rc}): $(tr '\n' ' ' <"$err_file" | tail -c 1500)" >&2
    cleanup_partial_clone
    return 1
  fi
  if [[ ! -d "${dest}/${subdir}" ]]; then
    # Some git versions set patterns but still need an explicit checkout.
    if ! git -C "$dest" checkout "$branch" >"$err_file" 2>&1; then
      rc=$?
      echo "git checkout after sparse-checkout failed (exit ${rc}): $(tr '\n' ' ' <"$err_file" | tail -c 1500)" >&2
      cleanup_partial_clone
      return 1
    fi
  fi
  if [[ ! -d "${dest}/${subdir}" ]]; then
    echo "sparse-checkout succeeded but path missing: ${dest}/${subdir} (check GitHub URL subdir)" >&2
    cleanup_partial_clone
    return 1
  fi
  git -C "$dest" remote set-url origin "$clone_url" >/dev/null 2>&1 || true
  rm -f "$err_file"
  printf '%s\n' "${dest}/${subdir}"
}

# Resolve submission dir from an existing sparse clone (URL subdir only).
resolve_existing_submission_dir() {
  local repo_dir="$1"
  local github_url="$2"
  local subdir

  if ! [[ -d "$repo_dir" ]]; then
    return 1
  fi

  mapfile -t _gh < <(parse_github "$github_url")
  subdir="${_gh[2]}"
  if [[ -n "$subdir" && -d "${repo_dir}/${subdir}" ]]; then
    printf '%s\n' "${repo_dir}/${subdir}"
    return 0
  fi
  return 1
}

wait_for_vllm() {
  local port="$1"
  local pid="${2:-}"
  local elapsed=0

  echo "[vllm] Waiting for server on port ${port} ..." >&2
  until curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[vllm] Process ${pid} exited before becoming ready." >&2
      return 1
    fi
    if (( elapsed >= VLLM_READY_TIMEOUT_SEC )); then
      echo "[vllm] Timed out after ${VLLM_READY_TIMEOUT_SEC}s." >&2
      return 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  echo "[vllm] Server is ready." >&2
}

stop_vllm() {
  local pid="${1:-}"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  echo "[vllm] Stopping PID ${pid} ..."

  # Prefer process-group kill only when this PID is the group leader (e.g. started
  # via setsid). Otherwise only signal the PID so we don't kill the eval shell.
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
    echo "[vllm] PID ${pid} is process-group leader; signaling group."
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  else
    # Recursively terminate children (EngineCore, etc.), then the parent.
    local child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
      stop_vllm "$child"
    done
    kill -TERM "$pid" 2>/dev/null || true
  fi

  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      echo "[vllm] PID ${pid} exited."
      return 0
    fi
    sleep 1
  done

  echo "[vllm] PID ${pid} still alive; sending SIGKILL ..."
  if [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  else
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
      kill -KILL "$child" 2>/dev/null || true
    done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

# Stop any vLLM process serving this model path (state file + /proc scan).
wind_down_vllm_for_work() {
  local work="$1"
  local model_path="${work}/decompressed_model"
  local state_file="${work}/vllm.state"
  local pid="" port="" stated_model=""

  if [[ -f "$state_file" ]]; then
    pid="$(read_vllm_state_field "$state_file" pid || true)"
    port="$(read_vllm_state_field "$state_file" port || true)"
    stated_model="$(read_vllm_state_field "$state_file" model || true)"
  fi

  if [[ -n "${VLLM_PID:-}" ]]; then
    stop_vllm "$VLLM_PID"
    VLLM_PID=""
  fi

  if [[ -n "$pid" ]]; then
    stop_vllm "$pid"
  fi

  # Catch orphans / reused servers whose PID wasn't tracked.
  if [[ -d "$model_path" ]] || [[ -n "$stated_model" ]]; then
    local target found orphan_pid
    target="$(readlink -f "${stated_model:-$model_path}" 2>/dev/null || echo "${stated_model:-$model_path}")"
    found="$(find_vllm_process_for_model "$target" || true)"
    if [[ -n "$found" ]]; then
      orphan_pid="${found%% *}"
      echo "[vllm] Found leftover server PID=${orphan_pid} for ${target}; stopping."
      stop_vllm "$orphan_pid"
    fi
  fi

  if [[ -n "$port" ]] && curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "[vllm] WARNING: port ${port} still responds after wind-down." >&2
    return 1
  fi

  rm -f "$state_file"
  echo "[vllm] Wind-down complete for ${work}."
}

# Stop vLLM first, then remove cloned repo + model artifacts.
cleanup_submission_artifacts() {
  local work="$1"
  echo "[cleanup] Winding down vLLM (if any) before deleting artifacts ..."
  wind_down_vllm_for_work "$work" || true

  echo "[cleanup] Removing repo and model dirs under ${work} ..."
  rm -rf \
    "${work}/repo" \
    "${work}/compressed_model" \
    "${work}/decompressed_model" \
    "${work}/vllm.state" \
    "${work}"/eval_config.*.yaml
}

write_vllm_state() {
  local state_file="$1"
  local pid="$2"
  local port="$3"
  local model="$4"
  local max_model_len="${5:-$MAX_MODEL_LEN}"
  cat >"$state_file" <<EOF
pid=${pid}
port=${port}
model=${model}
max_model_len=${max_model_len}
EOF
}

read_vllm_state_field() {
  local state_file="$1"
  local key="$2"
  [[ -f "$state_file" ]] || return 1
  sed -n "s/^${key}=//p" "$state_file" | head -n 1
}

# True if PID is alive and its cmdline references model_path (absolute).
pid_serves_model() {
  local pid="$1"
  local model_path="$2"
  local cmdline

  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  # Must look like a vLLM API server and reference this checkpoint path.
  [[ "$cmdline" == *vllm* ]] || [[ "$cmdline" == *api_server* ]] || return 1
  [[ "$cmdline" == *"${model_path}"* ]] || return 1
  return 0
}

# Scan for an existing vLLM process whose --model matches model_path.
# Prints: pid port   (port may be empty if not parseable)
find_vllm_process_for_model() {
  local model_path="$1"
  "$PYTHON" - "$model_path" <<'PY'
import os, re, sys
from pathlib import Path

model = str(Path(sys.argv[1]).resolve())
proc = Path("/proc")
for entry in proc.iterdir():
    if not entry.name.isdigit():
        continue
    cmdline_path = entry / "cmdline"
    try:
        raw = cmdline_path.read_bytes()
    except OSError:
        continue
    if not raw:
        continue
    parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
    text = " ".join(parts)
    if "vllm" not in text and "api_server" not in text:
        continue
    if model not in text:
        continue
    port = ""
    for i, p in enumerate(parts):
        if p in ("--port", "-p") and i + 1 < len(parts):
            port = parts[i + 1]
            break
        m = re.match(r"--port=(.+)", p)
        if m:
            port = m.group(1)
            break
    print(f"{entry.name} {port}".rstrip())
    break
PY
}

vllm_health_ok() {
  local port="$1"
  curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null
}

# Log helper: write to run_log + stderr so stdout can carry the port only.
_vllm_log() {
  local run_log="$1"
  shift
  echo "$@" | tee -a "$run_log" >&2
}

# Reuse a live server for model_path if possible; else start one.
# Sets VLLM_PID, REUSED_VLLM (0/1). Prints only the port on stdout.
ensure_vllm_server() {
  local model_path="$1"
  local state_file="$2"
  local vllm_log="$3"
  local run_log="$4"
  local max_model_len="${MAX_MODEL_LEN}"

  model_path="$(readlink -f "$model_path")"
  local pid="" port="" stated_len=""

  # 1) Prefer saved state from a previous run (same model + same max_model_len).
  if [[ -f "$state_file" ]]; then
    pid="$(read_vllm_state_field "$state_file" pid || true)"
    port="$(read_vllm_state_field "$state_file" port || true)"
    stated_len="$(read_vllm_state_field "$state_file" max_model_len || true)"
    local stated_model
    stated_model="$(read_vllm_state_field "$state_file" model || true)"
    if [[ -n "$pid" && -n "$port" && -n "$stated_model" ]] \
      && [[ "$(readlink -f "$stated_model" 2>/dev/null || echo "$stated_model")" == "$model_path" ]] \
      && [[ "$stated_len" == "$max_model_len" ]] \
      && pid_serves_model "$pid" "$model_path" \
      && vllm_health_ok "$port"; then
      _vllm_log "$run_log" "[vllm] Reusing existing server PID=${pid} port=${port} max_model_len=${max_model_len} (state file)."
      VLLM_PID="$pid"
      REUSED_VLLM=1
      printf '%s\n' "$port"
      return 0
    fi
  fi

  # 2) Scan /proc for a matching vLLM process with the same --max-model-len.
  local found cmdline
  found="$(find_vllm_process_for_model "$model_path" || true)"
  if [[ -n "$found" ]]; then
    pid="${found%% *}"
    port="${found#* }"
    if [[ "$port" == "$pid" ]]; then
      port=""
    fi
    if [[ -n "$pid" ]] && pid_serves_model "$pid" "$model_path"; then
      cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
      if [[ "$cmdline" == *"--max-model-len ${max_model_len}"* || "$cmdline" == *"--max-model-len=${max_model_len}"* ]] \
        && [[ -n "$port" ]] && vllm_health_ok "$port"; then
        _vllm_log "$run_log" "[vllm] Reusing existing server PID=${pid} port=${port} max_model_len=${max_model_len} (process scan)."
        write_vllm_state "$state_file" "$pid" "$port" "$model_path" "$max_model_len"
        VLLM_PID="$pid"
        REUSED_VLLM=1
        printf '%s\n' "$port"
        return 0
      fi
    fi
  fi

  # 3) Launch a new server.
  _vllm_log "$run_log" "[vllm] Starting new server (max_model_len=${max_model_len} = 1000+${MAX_NEW_TOKENS}) ..."
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
  # FlashInfer JIT shells out to `ninja`; keep venv bin on PATH for setsid children.
  export PATH="${SCRIPT_DIR}/.venv/bin:${PATH}"
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

  port="$(find_free_port)"
  # setsid → own process group so cleanup can signal the whole tree safely.
  setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len "$max_model_len" \
    --language-model-only \
    --port "$port" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    >"$vllm_log" 2>&1 &
  VLLM_PID=$!
  REUSED_VLLM=0
  write_vllm_state "$state_file" "$VLLM_PID" "$port" "$model_path" "$max_model_len"
  _vllm_log "$run_log" "  vLLM PID=${VLLM_PID} port=${port} max_model_len=${max_model_len} (log: ${vllm_log})"

  if ! wait_for_vllm "$port" "$VLLM_PID"; then
    echo "----- vLLM log (tail) -----" | tee -a "$run_log" >&2
    tail -n 100 "$vllm_log" | tee -a "$run_log" >&2 || true
    return 1
  fi
  printf '%s\n' "$port"
}

# Remove cloned repo + downloaded/decompressed model artifacts for a submission.
# (vLLM wind-down happens in the earlier cleanup_submission_artifacts definition.)

already_succeeded() {
  local roll="$1"
  "$PYTHON" - "$RESULTS_CSV" "$roll" <<'PY'
import csv, sys
from pathlib import Path

path, roll = Path(sys.argv[1]), sys.argv[2]
if not path.exists():
    raise SystemExit(1)
ok = False
with path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if (row.get("roll") or "").strip() == roll and (row.get("status") or "").strip() == "ok":
            ok = True
raise SystemExit(0 if ok else 1)
PY
}

# Grep Err/err lines from eval (stdout) + vLLM logs for Slack.
collect_error_snippets() {
  local run_log="$1"
  local vllm_log="$2"
  local snippet=""

  for f in "$run_log" "$vllm_log"; do
    if [[ -f "$f" ]]; then
      snippet+="--- $(basename "$f") ---"$'\n'
      snippet+="$(grep -e Err -e err -e Error -e ERROR -e Traceback -e Exception "$f" 2>/dev/null | tail -n 100 || true)"$'\n'
    fi
  done
  printf '%s' "$snippet"
}

collect_vllm_err_snippets() {
  local vllm_log="$1"
  if [[ ! -f "$vllm_log" ]]; then
    printf '%s' "(no vLLM log at ${vllm_log})"
    return 0
  fi
  local snippet
  snippet="$(grep -e Err -e err -e Error -e ERROR -e Traceback -e Exception "$vllm_log" 2>/dev/null | tail -n 80 || true)"
  if [[ -z "$snippet" ]]; then
    # Fall back to last lines so Slack still has signal when grep misses.
    snippet="$(tail -n 40 "$vllm_log" 2>/dev/null || true)"
  fi
  printf '%s' "$snippet"
}

# Slack-facing dataset label (basename of DATASET).
slack_dataset_label() {
  local ds="${1:-$DATASET}"
  printf '%s\n' "$(basename "$ds")"
}

notify_slack_eval() {
  local status="$1"
  local name="$2"
  local roll="$3"
  local reason="$4"
  local run_log="$5"
  local vllm_log="$6"
  local accuracy="${7:-}"
  local failed_step="${8:-}"
  local size_frac="${9:-}"
  local ds_label
  ds_label="$(slack_dataset_label)"

  local line1 line2 message
  if [[ "$status" == "ok" ]]; then
    line1="✅ ${name} (${roll}) | ${ds_label} | accuracy=${accuracy:-n/a} | size_frac=${size_frac:-n/a}"
    line2="status=ok"
  else
    line1="❌ ${name} (${roll}) | ${ds_label} | step=${failed_step:-unknown} | size_frac=${size_frac:-n/a}"
    line2="${reason:-failed}"
  fi
  message="${line1}"$'\n'"${line2}"

  # Only attach grepped Err lines from the vLLM log on vLLM failures.
  if [[ "$status" != "ok" && "$failed_step" == "5_vllm" ]]; then
    local vllm_errs
    vllm_errs="$(collect_vllm_err_snippets "$vllm_log")"
    message+=$'\n'"\`\`\`"$'\n'"${vllm_errs:-"(no Err matches in vLLM log)"}"$'\n'"\`\`\`"
  elif [[ "${SLACK_LOG}" -eq 1 ]]; then
    local err_dump
    err_dump="$(collect_error_snippets "$run_log" "$vllm_log")"
    message+=$'\n'"\`\`\`"$'\n'"${err_dump:-"(no Err/err matches)"}"$'\n'"\`\`\`"
  fi

  # Never fail the eval pipeline because Slack is down.
  set +e
  printf '%s\n' "$message" | "$PYTHON" "${SCRIPT_DIR}/slack.py" -
  set -e
}

append_result() {
  "$PYTHON" - "$RESULTS_CSV" "$@" <<'PY'
import csv
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
row = {
    "roll": sys.argv[2],
    "name": sys.argv[3],
    "email": sys.argv[4],
    "compression_target": sys.argv[5],
    "github_url": sys.argv[6],
    "huggingface_url": sys.argv[7],
    "accuracy": sys.argv[8],
    "parse_rate": sys.argv[9],
    "num_correct": sys.argv[10],
    "num_examples": sys.argv[11],
    "size_frac": sys.argv[12],
    "status": sys.argv[13],
    "error": sys.argv[14],
    "output_json": sys.argv[15],
}
fieldnames = list(row.keys())
exists = out_path.exists() and out_path.stat().st_size > 0
with out_path.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
    w.writerow(row)
PY
}

json_field() {
  "$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$1" "$2"
}

# Globals updated by evaluate_one.
VLLM_PID=""
REUSED_VLLM=0
LAST_ACCURACY=""
LAST_PARSE_RATE=""
LAST_NUM_CORRECT=""
LAST_NUM_EXAMPLES=""
LAST_SIZE_FRAC=""
LAST_FAILED_STEP=""
LAST_FAILURE_REASON=""

# Do not kill vLLM on EXIT — leave it running so a restart can reuse it.
cleanup() { :; }
trap cleanup EXIT

evaluate_one() {
  local github_url="$1"
  local huggingface_url="$2"
  local work="$3"
  local out_json="$4"
  local vllm_log="$5"
  local run_log="$6"

  local repo_dir="${work}/repo"
  local compressed_dir="${work}/compressed_model"
  local decompressed_dir="${work}/decompressed_model"
  local state_file="${work}/vllm.state"
  local submission_dir hf_id port tmp_cfg DECOMPRESS_PY
  local step_err

  LAST_ACCURACY=""
  LAST_PARSE_RATE=""
  LAST_NUM_CORRECT=""
  LAST_NUM_EXAMPLES=""
  LAST_SIZE_FRAC=""
  LAST_FAILED_STEP=""
  LAST_FAILURE_REASON=""
  VLLM_PID=""
  REUSED_VLLM=0

  fail_at() {
    local step="$1"
    local reason="$2"
    LAST_FAILED_STEP="$step"
    LAST_FAILURE_REASON="$reason"
    echo "[FAIL] step=${step}: ${reason}" | tee -a "$run_log" >&2
    return 1
  }

  # Run a command, tee output to run_log, return its real exit code (PIPESTATUS[0]).
  run_logged() {
    set +o pipefail
    "$@" 2>&1 | tee -a "$run_log"
    local rc=${PIPESTATUS[0]}
    set -o pipefail
    return "$rc"
  }

  # Capture stderr from a command that prints its primary result on stdout.
  # Usage: out="$(capture_cmd_stdout err_var cmd args...)" ; rc=$?
  # On failure, err_var holds stderr text.
  capture_cmd_stdout() {
    local _err_var="$1"
    shift
    local _out_file _err_file _rc _out
    _out_file="$(mktemp)"
    _err_file="$(mktemp)"
    "$@" >"$_out_file" 2>"$_err_file"
    _rc=$?
    _out="$(cat "$_out_file")"
    # shellcheck disable=SC2034
    printf -v "$_err_var" '%s' "$(cat "$_err_file")"
    rm -f "$_out_file" "$_err_file"
    printf '%s' "$_out"
    return "$_rc"
  }

  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
  HF_CLI="${SCRIPT_DIR}/.venv/bin/hf"

  # ---------- 1) Clone (never wipe an existing successful repo) ----------
  if [[ -d "$repo_dir" ]]; then
    echo "[1/6] Found existing repo dir: ${repo_dir}" | tee -a "$run_log"
    if [[ ! -d "${repo_dir}/.git" ]]; then
      fail_at "1_clone" "existing path ${repo_dir} is not a git repo (incomplete/failed prior clone); remove it manually and re-run"
      return 1
    fi
    if ! submission_dir="$(resolve_existing_submission_dir "$repo_dir" "$github_url")"; then
      fail_at "1_clone" "existing repo at ${repo_dir} could not be resolved to a submission subdir"
      return 1
    fi
    echo "[1/6] Reusing existing clone at: ${submission_dir}" | tee -a "$run_log"
  else
    echo "[1/6] Sparse-cloning submission folder ..." | tee -a "$run_log"
    step_err=""
    if ! submission_dir="$(capture_cmd_stdout step_err clone_submission "$github_url" "$repo_dir")"; then
      # clone_submission removes partial dest on failure
      if [[ -n "$step_err" ]]; then
        fail_at "1_clone" "$step_err"
      else
        fail_at "1_clone" "git clone failed for ${github_url}"
      fi
      return 1
    fi
    if [[ -z "$submission_dir" || ! -d "$submission_dir" ]]; then
      fail_at "1_clone" "clone reported success but submission dir missing: '${submission_dir}'"
      return 1
    fi
    echo "  Clone path: ${submission_dir}" | tee -a "$run_log"
  fi

  if [[ ! -f "${submission_dir}/decompress.py" ]]; then
    fail_at "1_clone" "sparse clone at ${submission_dir} is missing decompress.py"
    return 1
  fi
  echo "  Using decompress.py in: ${submission_dir}" | tee -a "$run_log"

  # Required layout under SubmissionNN/ (course tree).
  local missing_layout=()
  local req
  for req in compress.py decompress.py pyproject.toml README.md; do
    if [[ ! -f "${submission_dir}/${req}" ]]; then
      missing_layout+=("${req} (file)")
    fi
  done
  for req in compression decompression; do
    if [[ ! -d "${submission_dir}/${req}" ]]; then
      missing_layout+=("${req}/ (directory)")
    fi
  done
  for req in compression/__init__.py decompression/__init__.py; do
    if [[ ! -f "${submission_dir}/${req}" ]]; then
      missing_layout+=("${req} (file)")
    fi
  done
  if [[ ${#missing_layout[@]} -gt 0 ]]; then
    fail_at "0_format" "submission layout incomplete under ${submission_dir}; missing: ${missing_layout[*]}"
    return 1
  fi
  echo "  Submission layout OK (compress.py, decompress.py, pyproject.toml, README.md, compression/, decompression/)" | tee -a "$run_log"

  # ---------- 2) Download (skip if checkpoint already on disk) ----------
  if ! hf_id="$(capture_cmd_stdout step_err parse_hf_repo_id "$huggingface_url")"; then
    if [[ -n "$step_err" ]]; then
      fail_at "2_download" "$step_err"
    else
      fail_at "2_download" "invalid HuggingFace URL: ${huggingface_url}"
    fi
    return 1
  fi
  if has_model_checkpoint "$compressed_dir" || { dir_nonempty "$compressed_dir" && [[ -f "${compressed_dir}/.download_complete" ]]; }; then
    echo "[2/6] Reusing existing download at: ${compressed_dir}" | tee -a "$run_log"
  else
    echo "[2/6] Downloading HuggingFace model ..." | tee -a "$run_log"
    mkdir -p "$compressed_dir"
    if ! run_logged "$HF_CLI" download "$hf_id" --local-dir "$compressed_dir"; then
      fail_at "2_download" "hf download failed for ${hf_id} (see ${run_log})"
      return 1
    fi
    touch "${compressed_dir}/.download_complete"
  fi

  # Measure compressed text size vs original text tower (hardcoded).
  echo "[2/6] Measuring compressed checkpoint text size ..." | tee -a "$run_log"
  local measure_json compressed_text_gb
  if ! measure_json="$("$PYTHON" "${SCRIPT_DIR}/measure_checkpoint_bits.py" --json "$compressed_dir" 2>>"$run_log")"; then
    fail_at "2_download" "measure_checkpoint_bits.py failed on ${compressed_dir} (see ${run_log})"
    return 1
  fi
  compressed_text_gb="$("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["text_gb"])' "$measure_json")"
  LAST_SIZE_FRAC="$("$PYTHON" -c 'import sys; print(f"{float(sys.argv[1])/float(sys.argv[2]):.6f}")' "$compressed_text_gb" "$ORIGINAL_TEXT_GB")"
  echo "  compressed text=${compressed_text_gb} GB  original text=${ORIGINAL_TEXT_GB} GB  size_frac=${LAST_SIZE_FRAC}" | tee -a "$run_log"

  # Zero visual weights in the downloaded checkpoint before student decompress.
  echo "[2/6] Zeroing visual tensors in-place (${compressed_dir}) ..." | tee -a "$run_log"
  if [[ -f "${compressed_dir}/.visual_zero_complete" ]]; then
    echo "  Reusing previous ensure_visual_zero.py run" | tee -a "$run_log"
  else
    if ! run_logged "$PYTHON" "${SCRIPT_DIR}/ensure_visual_zero.py" "$compressed_dir"; then
      fail_at "2_download" "ensure_visual_zero.py failed on ${compressed_dir} (see ${run_log})"
      return 1
    fi
    touch "${compressed_dir}/.visual_zero_complete"
    # Compressed weights changed — force decompress to run again.
    rm -f "${decompressed_dir}/.decompress_complete"
  fi

  # ---------- 3) venv ----------
  echo "[3/6] Ensuring local .venv based on pyproject.toml ..." | tee -a "$run_log"
  if [[ -f "${submission_dir}/pyproject.toml" ]]; then
    if ! run_logged bash -lc "cd \"${submission_dir}\" && uv sync"; then
      fail_at "3_venv" "uv sync failed in student submission dir (see ${run_log})"
      return 1
    fi
    DECOMPRESS_PY="${submission_dir}/.venv/bin/python"
    if [[ ! -x "$DECOMPRESS_PY" ]]; then
      DECOMPRESS_PY="$PYTHON"
    fi
  else
    if ! run_logged bash -lc "cd \"${SCRIPT_DIR}\" && uv sync --extra cuda129"; then
      fail_at "3_venv" "uv sync --extra cuda129 failed in course root (see ${run_log})"
      return 1
    fi
    DECOMPRESS_PY="$PYTHON"
  fi

  # ---------- 4) Decompress (skip if output checkpoint exists) ----------
  if has_model_checkpoint "$decompressed_dir" || [[ -f "${decompressed_dir}/.decompress_complete" ]]; then
    echo "[4/6] Reusing existing decompressed model at: ${decompressed_dir}" | tee -a "$run_log"
  else
    echo "[4/6] Decompressing model ..." | tee -a "$run_log"
    mkdir -p "$decompressed_dir"
    if ! run_logged bash -lc "cd \"${submission_dir}\" && \"${DECOMPRESS_PY}\" decompress.py --model_name \"${MODEL_NAME_DECOMPRESS}\" --checkpoint_path \"${compressed_dir}\" --output_path \"${decompressed_dir}\""; then
      fail_at "4_decompress" "decompress.py failed (see ${run_log})"
      return 1
    fi
    if ! has_model_checkpoint "$decompressed_dir"; then
      fail_at "4_decompress" "decompress.py exited 0 but no checkpoint found under ${decompressed_dir}"
      return 1
    fi
    touch "${decompressed_dir}/.decompress_complete"
  fi

  decompressed_dir="$(readlink -f "$decompressed_dir")"

  # ---------- 5) vLLM (reuse if correct model already served) ----------
  echo "[5/6] Ensuring vLLM server for ${decompressed_dir} ..." | tee -a "$run_log"
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"

  step_err=""
  if ! port="$(capture_cmd_stdout step_err ensure_vllm_server "$decompressed_dir" "$state_file" "$vllm_log" "$run_log")"; then
    # Prefer last Error/Traceback line from vLLM log when available.
    local vllm_hint=""
    if [[ -f "$vllm_log" ]]; then
      vllm_hint="$(grep -E 'Error|Exception|Traceback|failed' "$vllm_log" 2>/dev/null | tail -n 3 | tr '\n' ' ' || true)"
    fi
    if [[ -n "$vllm_hint" ]]; then
      fail_at "5_vllm" "${vllm_hint} (see ${vllm_log})"
    elif [[ -n "$step_err" ]]; then
      fail_at "5_vllm" "${step_err} (see ${vllm_log})"
    else
      fail_at "5_vllm" "vLLM failed to start or become ready (see ${vllm_log})"
    fi
    return 1
  fi
  if [[ -z "$port" ]]; then
    fail_at "5_vllm" "vLLM returned empty port"
    return 1
  fi

  # ---------- 6) Eval ----------
  echo "[6/6] Running evaluation ..." | tee -a "$run_log"
  tmp_cfg="$(mktemp "${work}/eval_config.XXXXXX.yaml")"
  if ! "$PYTHON" - "$CONFIG" "$tmp_cfg" "$out_json" "$port" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, out_json, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
cfg["output"] = out_json
cfg["vllm_base_url"] = f"http://localhost:{port}/v1"
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  then
    fail_at "6_eval" "failed to write temporary eval config"
    return 1
  fi

  if ! run_logged "$PYTHON" evaluation/run_eval.py \
    --config "$tmp_cfg" \
    --dataset "$DATASET" \
    --port "$port" \
    --limit "$LIMIT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-concurrency "$MAX_CONCURRENCY"; then
    fail_at "6_eval" "run_eval.py failed (see ${run_log})"
    return 1
  fi

  if [[ ! -f "$out_json" ]]; then
    fail_at "6_eval" "missing output JSON ${out_json}"
    return 1
  fi

  if ! read -r LAST_ACCURACY LAST_PARSE_RATE LAST_NUM_CORRECT LAST_NUM_EXAMPLES < <(
    "$PYTHON" - "$out_json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
s = data["summary"]
print(s["accuracy"], s["parse_rate"], s["num_correct"], s["num_examples"])
PY
  ); then
    fail_at "6_eval" "failed to parse summary from ${out_json}"
    return 1
  fi

  LAST_FAILED_STEP=""
  LAST_FAILURE_REASON=""
}

# Load submissions as JSON lines (handles quoted CSV fields).
mapfile -t SUBMISSIONS < <(
  "$PYTHON" - "$CSV_IN" <<'PY'
import csv
import json
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        norm = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        payload = {
            "email": norm.get("Email Address", ""),
            "name": norm.get("Name", ""),
            "roll": norm.get("Roll no.", norm.get("Roll no", "")),
            "compression_target": norm.get("Compression Target", ""),
            "github_url": norm.get("GitHub URL", ""),
            "huggingface_url": norm.get("HuggingFace URL", ""),
        }
        if not payload["github_url"] or not payload["huggingface_url"]:
            continue
        print(json.dumps(payload, ensure_ascii=False))
PY
)

echo "Found ${#SUBMISSIONS[@]} submissions in ${CSV_IN}"
echo "Results → ${RESULTS_CSV}"
echo

if [[ ${#SUBMISSIONS[@]} -eq 0 ]]; then
  echo "ERROR: No submissions with both GitHub and HuggingFace URLs in ${CSV_IN}." >&2
  exit 1
fi

echo "[0/6] Validating GitHub path + HuggingFace naming for ${WEEK_LABEL} ..."
FORMAT_REPORT=""
FORMAT_RC=0
set +e
FORMAT_REPORT="$(validate_submission_formats "$WEEK_LABEL" "${SUBMISSIONS[@]}" 2>&1)"
FORMAT_RC=$?
set -e
printf '%s\n' "$FORMAT_REPORT"
FORMAT_ROSTER="$(format_students_roster "${SUBMISSIONS[@]}")"
if [[ $FORMAT_RC -ne 0 ]]; then
  echo "[FAIL] step=0_format: naming/folder-structure check failed — aborting." >&2
  notify_slack_format_failure "$WEEK_LABEL" "$FORMAT_REPORT" "$FORMAT_ROSTER"
  exit 1
fi
notify_slack_format_success "$WEEK_LABEL" "$FORMAT_REPORT" "$FORMAT_ROSTER" "${#SUBMISSIONS[@]}"
echo

FAILED_COUNT=0
OK_COUNT=0

for entry in "${SUBMISSIONS[@]}"; do
  roll="$(json_field "$entry" roll)"
  name="$(json_field "$entry" name)"
  email="$(json_field "$entry" email)"
  compression_target="$(json_field "$entry" compression_target)"
  github_url="$(json_field "$entry" github_url)"
  huggingface_url="$(json_field "$entry" huggingface_url)"

  safe_roll="$(echo "$roll" | tr -c 'A-Za-z0-9._-' '_')"
  work="${WORK_ROOT}/${safe_roll}"
  out_json="outputs/${safe_roll}_sample_math.json"
  vllm_log="logs/vllm_${safe_roll}.log"
  run_log="logs/eval_${safe_roll}.log"

  if already_succeeded "$roll"; then
    echo "Skipping ${roll} (${name}) — already status=ok in ${RESULTS_CSV}"
    echo
    continue
  fi

  mkdir -p "$work"

  # Fresh logs each attempt so Slack/grep never mixes in previous errors.
  : >"$run_log"
  : >"$vllm_log"
  {
    echo "======== $(date -Is) start ========"
    echo "roll=${roll} name=${name}"
    echo "dataset=${DATASET}"
    echo "interact=${INTERACT}"
  } >>"$run_log"

  echo "============================================================"
  echo "Evaluating: ${name} (${roll})"
  echo "  GitHub : ${github_url}"
  echo "  HF     : ${huggingface_url}"
  echo "  Workdir: ${work}"
  echo "============================================================"

  status="ok"
  err=""
  LAST_FAILED_STEP=""
  LAST_FAILURE_REASON=""
  LAST_SIZE_FRAC=""

  set +e
  evaluate_one "$github_url" "$huggingface_url" "$work" "$out_json" "$vllm_log" "$run_log"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    status="fail"
    if [[ -n "${LAST_FAILURE_REASON}" ]]; then
      err="step=${LAST_FAILED_STEP:-unknown}: ${LAST_FAILURE_REASON}"
    else
      err="step=${LAST_FAILED_STEP:-unknown}: see ${run_log} and ${vllm_log}"
    fi
    FAILED_COUNT=$((FAILED_COUNT + 1))

    append_result \
      "$roll" "$name" "$email" "$compression_target" \
      "$github_url" "$huggingface_url" \
      "${LAST_ACCURACY}" "${LAST_PARSE_RATE}" \
      "${LAST_NUM_CORRECT}" "${LAST_NUM_EXAMPLES}" \
      "${LAST_SIZE_FRAC}" \
      "$status" "$err" "$out_json"

    notify_slack_eval \
      "$status" "$name" "$roll" \
      "$err" "$run_log" "$vllm_log" \
      "${LAST_ACCURACY}" \
      "${LAST_FAILED_STEP:-unknown}" \
      "${LAST_SIZE_FRAC}"

    echo "Recorded ${roll} → status=fail step=${LAST_FAILED_STEP:-unknown} size_frac=${LAST_SIZE_FRAC}"

    if [[ "$INTERACT" -eq 1 ]]; then
      echo "[interact] Stopping at first failure."
      echo "Artifacts kept under ${work} (vLLM left running if up)."
      echo "Re-run: ./eval.sh --week ${WEEK_NUM} ${CSV_IN} ${RESULTS_CSV} --interact"
      exit 1
    fi

    # Batch mode: free GPU and remove on-disk clone/model artifacts.
    echo "[batch] Cleaning up artifacts under ${work}"
    cleanup_submission_artifacts "$work"
    VLLM_PID=""
    echo
    continue
  fi

  OK_COUNT=$((OK_COUNT + 1))

  # Success: wind down vLLM, free disk, record metrics.
  append_result \
    "$roll" "$name" "$email" "$compression_target" \
    "$github_url" "$huggingface_url" \
    "${LAST_ACCURACY}" "${LAST_PARSE_RATE}" \
    "${LAST_NUM_CORRECT}" "${LAST_NUM_EXAMPLES}" \
    "${LAST_SIZE_FRAC}" \
    "$status" "$err" "$out_json"

  notify_slack_eval \
    "$status" "$name" "$roll" \
    "evaluation completed successfully" "$run_log" "$vllm_log" \
    "${LAST_ACCURACY}" \
    "" \
    "${LAST_SIZE_FRAC}"

  cleanup_submission_artifacts "$work"
  VLLM_PID=""

  echo "Recorded ${roll} → status=${status} accuracy=${LAST_ACCURACY} size_frac=${LAST_SIZE_FRAC}"
  echo
done

echo "Done. Results written to ${RESULTS_CSV}"
echo "Summary: ok=${OK_COUNT} fail=${FAILED_COUNT}"
if [[ "$FAILED_COUNT" -gt 0 ]]; then
  exit 1
fi
