#!/usr/bin/env bash
# scripts/lab_reset.sh — restore lab to ASL-only baseline between participants.
#
# Removes participant-generated artefacts (trained models, extracted landmarks,
# deployed Triton model) while leaving shared assets (isl_frames dataset,
# languages/isl config and references, Docker images) untouched.
#
# Usage:
#   bash scripts/lab_reset.sh           # normal reset
#   bash scripts/lab_reset.sh --dry-run # show what would be removed, do nothing

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
TRITON_HTTP="http://localhost:8010"
APP_HTTP="http://localhost:7860"
DRY_RUN=false

for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

log()    { printf '[reset] %s\n' "$*"; }
ok()     { printf '[reset] \e[32m✓\e[0m %s\n' "$*"; }
warn()   { printf '[reset] \e[33m⚠\e[0m  %s\n' "$*"; }
fail()   { printf '[reset] \e[31m✗\e[0m %s\n' "$*" >&2; exit 1; }

remove() {
    # remove() <path> — removes file or directory, honouring --dry-run
    local target="$1"
    if [[ -e "$target" || -d "$target" ]]; then
        if $DRY_RUN; then
            log "  [dry-run] would remove: $target"
        else
            rm -rf "$target"
            log "  removed: $target"
        fi
    fi
}

cd "$REPO"

if $DRY_RUN; then
    log "=== DRY RUN — no changes will be made ==="
fi

# ---------------------------------------------------------------------------
# 1. Stop tutor-app so it doesn't reload a half-reset language registry.
# ---------------------------------------------------------------------------
log "Stopping tutor-app..."
if $DRY_RUN; then
    log "  [dry-run] would run: docker compose stop tutor-app"
else
    docker compose stop tutor-app >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
# 2. Remove all non-ASL Triton model directories.
#    Poll mode will unload them within the next cycle (~5 s).
# ---------------------------------------------------------------------------
log "Removing non-baseline Triton models..."
for model_dir in "$REPO/triton_repo"/*/; do
    [[ -d "$model_dir" ]] || continue
    name=$(basename "$model_dir")
    if [[ "$name" != "asl_classifier" ]]; then
        remove "$model_dir"
    fi
done

# ---------------------------------------------------------------------------
# 3. Remove participant-generated artefacts.
#    Shared assets (isl_frames, languages/isl config + references) are kept.
# ---------------------------------------------------------------------------
log "Removing participant artefacts..."
remove "$REPO/checkpoints/isl"
remove "$REPO/datasets/isl_landmarks.csv"
remove "$REPO/datasets/isl_landmarks_combined.csv"
remove "$REPO/datasets/self_capture/isl"
remove "$REPO/languages/isl/model.onnx"

# ---------------------------------------------------------------------------
# 4. Restart tutor-app.
# ---------------------------------------------------------------------------
log "Restarting tutor-app..."
if $DRY_RUN; then
    log "  [dry-run] would run: docker compose up -d tutor-app"
else
    docker compose up -d tutor-app >/dev/null
fi

# ---------------------------------------------------------------------------
# 5. Verify baseline is healthy.
# ---------------------------------------------------------------------------
if $DRY_RUN; then
    log "=== DRY RUN complete — no changes made ==="
    exit 0
fi

log "Waiting for services..."
sleep 8

# Check ASL classifier is ready
asl_ready=false
for i in {1..12}; do
    if curl -sf "$TRITON_HTTP/v2/models/asl_classifier/ready" >/dev/null 2>&1; then
        asl_ready=true
        break
    fi
    sleep 2
done

$asl_ready || fail "asl_classifier did not become READY within 30 s — check: docker logs sign-tutor-triton-1"
ok "Triton: asl_classifier READY"

# Check ISL classifier is gone
if curl -sf "$TRITON_HTTP/v2/models/isl_classifier/ready" >/dev/null 2>&1; then
    warn "isl_classifier is still READY — Triton may not have polled yet; wait 10 s and retry"
else
    ok "Triton: isl_classifier not present"
fi

# Check tutor-app is responding
if curl -sf "$APP_HTTP/" >/dev/null 2>&1; then
    ok "tutor-app responding on :7860"
else
    fail "tutor-app not responding on :7860 — check: docker logs sign-tutor-tutor-app-1"
fi

log "Reset complete. Lab is at ASL-only baseline."
