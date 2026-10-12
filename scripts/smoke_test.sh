#!/usr/bin/env bash
# Usage: scripts/smoke_test.sh [base_url]   (exit code != 0 on any failure)
set -euo pipefail
BASE="${1:-http://localhost:8000}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "-> GET $BASE/health"
curl -fsS "$BASE/health"; echo
echo "-> POST $BASE/predict"
RESP=$(curl -fsS -X POST "$BASE/predict" -H "Content-Type: application/json" \
        -d @"$DIR/sample_input.json")
echo "$RESP"
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); assert r['prediction'] in (0,1); assert 0.5<=r['confidence']<=1.0; print('OK: response schema valid')"
echo "-> GET $BASE/metrics (first lines)"
curl -fsS "$BASE/metrics" | grep -E "^(http_requests_total|predictions_total)" | head -5
echo "Smoke test passed."
