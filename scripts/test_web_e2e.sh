#!/usr/bin/env bash
# PrintForge Web API end-to-end smoke test
# Usage: bash scripts/test_web_e2e.sh
set -euo pipefail

HOST="${PRINTFORGE_HOST:-http://127.0.0.1:8000}"
DEMO_IMAGE="${PRINTFORGE_DEMO_IMAGE:-examples/demo_cup.png}"
PASS=0
FAIL=0
TMPDIR="$(mktemp -d)"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

report() {
    local name="$1" code="$2"
    if [ "$code" -eq 0 ]; then
        echo "  PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $name"
        FAIL=$((FAIL + 1))
    fi
}

# ── Start server ──────────────────────────────────────────────────
echo "Starting PrintForge server..."
python3 -m uvicorn printforge.server:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready (up to 15s)
for i in $(seq 1 30); do
    if curl -sf "$HOST/health" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

if ! curl -sf "$HOST/health" >/dev/null 2>&1; then
    echo "FAIL  Server did not start within 15s"
    exit 1
fi
echo "Server ready (PID $SERVER_PID)"
echo

# ── Tests ─────────────────────────────────────────────────────────
echo "Running e2e tests..."

# 1. Health check
STATUS=$(curl -sf -o /dev/null -w "%{http_code}" "$HOST/health")
[ "$STATUS" = "200" ]; report "/health" $?

# 2. Generate with placeholder backend
HTTP_CODE=$(curl -sf -o "$TMPDIR/placeholder.stl" -w "%{http_code}" \
    -F "image=@$DEMO_IMAGE" \
    "$HOST/api/generate?backend=placeholder&format=stl")
[ "$HTTP_CODE" = "200" ] && [ -s "$TMPDIR/placeholder.stl" ]
report "/api/generate?backend=placeholder" $?

# 3. Generate with auto backend (may fail if no quota — that's OK)
HTTP_CODE=$(curl -s -o "$TMPDIR/auto.stl" -w "%{http_code}" \
    -F "image=@$DEMO_IMAGE" \
    "$HOST/api/generate?backend=auto&format=stl" 2>/dev/null)
if [ "$HTTP_CODE" = "200" ] && [ -s "$TMPDIR/auto.stl" ]; then
    report "/api/generate?backend=auto" 0
else
    echo "  SKIP  /api/generate?backend=auto (HTTP $HTTP_CODE — expected without quota)"
fi

# 4. Quality endpoint
if [ -s "$TMPDIR/placeholder.stl" ]; then
    HTTP_CODE=$(curl -sf -o "$TMPDIR/quality.json" -w "%{http_code}" \
        -F "mesh=@$TMPDIR/placeholder.stl" \
        "$HOST/api/quality")
    [ "$HTTP_CODE" = "200" ]
    report "/api/quality" $?
else
    echo "  SKIP  /api/quality (no STL from previous step)"
fi

# 5. Cost endpoint
if [ -s "$TMPDIR/placeholder.stl" ]; then
    HTTP_CODE=$(curl -sf -o "$TMPDIR/cost.json" -w "%{http_code}" \
        -F "mesh=@$TMPDIR/placeholder.stl" \
        "$HOST/api/cost")
    [ "$HTTP_CODE" = "200" ]
    report "/api/cost" $?
else
    echo "  SKIP  /api/cost (no STL from previous step)"
fi

# 6. Health check (final)
STATUS=$(curl -sf -o /dev/null -w "%{http_code}" "$HOST/health")
[ "$STATUS" = "200" ]; report "/health (final)" $?

# ── Summary ───────────────────────────────────────────────────────
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
