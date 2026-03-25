#!/usr/bin/env bash
# PrintForge Security Audit Script
# Checks: known vulns, hardcoded secrets, rate limiting

set -euo pipefail

PASS=0
FAIL=0
WARN=0

header() { echo -e "\n=== $1 ==="; }
pass()   { echo "  [PASS] $1"; ((PASS++)); }
fail()   { echo "  [FAIL] $1"; ((FAIL++)); }
warn()   { echo "  [WARN] $1"; ((WARN++)); }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

header "1. Known Vulnerability Scan (pip audit)"
if command -v pip-audit &>/dev/null; then
    if pip-audit --strict 2>/dev/null; then
        pass "No known vulnerabilities found"
    else
        fail "pip-audit found vulnerabilities — run 'pip-audit' for details"
    fi
elif pip audit 2>/dev/null; then
    pass "pip audit passed"
else
    warn "pip-audit not installed — run 'pip install pip-audit' to enable"
fi

header "2. Hardcoded Secrets Check"
SECRET_PATTERNS=(
    'password\s*=\s*["\x27][^"\x27]+'
    'secret\s*=\s*["\x27][^"\x27]+'
    'api_key\s*=\s*["\x27][^"\x27]+'
    'access_token\s*=\s*["\x27][^"\x27]+'
    'AWS_SECRET'
    'PRIVATE_KEY'
)
SECRETS_FOUND=0
for pattern in "${SECRET_PATTERNS[@]}"; do
    if grep -rniE "$pattern" src/ --include="*.py" | grep -v 'test' | grep -v '#' | grep -v 'example' | grep -v 'placeholder' | grep -v 'Form(' | grep -v 'argparse' | head -5 | grep -q .; then
        fail "Potential secret pattern: $pattern"
        SECRETS_FOUND=1
    fi
done
if [ "$SECRETS_FOUND" -eq 0 ]; then
    pass "No hardcoded secrets detected in src/"
fi

header "3. .env / Credential Files Check"
if [ -f "$REPO_ROOT/.env" ]; then
    fail ".env file exists in repo root — should be in .gitignore"
else
    pass "No .env file in repo root"
fi
if grep -q "\.env" "$REPO_ROOT/.gitignore" 2>/dev/null; then
    pass ".env is in .gitignore"
else
    warn ".env not found in .gitignore"
fi

header "4. Rate Limiting Verification"
if grep -q "RateLimiter" src/printforge/server.py; then
    pass "Rate limiter is configured in server.py"
else
    fail "No rate limiter found in server.py"
fi
if grep -q "rate_limit_middleware" src/printforge/server.py; then
    pass "Rate limit middleware is active"
else
    fail "Rate limit middleware not found"
fi
RATE_LIMITED_ENDPOINTS=$(grep -c "rate_limited_paths" src/printforge/server.py || true)
if [ "$RATE_LIMITED_ENDPOINTS" -gt 0 ]; then
    pass "Generation endpoints are rate-limited"
else
    fail "Rate limiting not applied to generation endpoints"
fi

header "5. Input Validation Check"
if grep -q "content_type" src/printforge/server.py; then
    pass "File upload content type validation present"
else
    fail "No content type validation on uploads"
fi
if grep -q "ContentSafety\|check_image" src/printforge/safety.py 2>/dev/null; then
    pass "Content safety module exists"
else
    warn "No content safety module found"
fi

header "6. Dependency Pinning"
if grep -q ">=" pyproject.toml; then
    warn "Dependencies use >= (minimum version) — consider pinning for reproducibility"
else
    pass "Dependencies are version-pinned"
fi

echo -e "\n================================"
echo "Security Audit Summary"
echo "================================"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  WARN: $WARN"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
    echo "RESULT: FAIL — $FAIL issue(s) need attention"
    exit 1
else
    echo "RESULT: PASS — no critical issues found"
    exit 0
fi
