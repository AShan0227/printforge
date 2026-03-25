#!/usr/bin/env python3
"""
PrintForge Launch Day Monitor
==============================
Checks GitHub stars, issues, and PyPI downloads.
Run every hour on launch day: watch -n 3600 python3 scripts/launch_monitor.py
"""

import json
import subprocess
import sys
from datetime import datetime


REPO = "AShan0227/printforge"
PYPI_PACKAGE = "printforge"


def run_cmd(cmd: list[str], timeout: int = 15) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_github_stats() -> dict:
    """Fetch repo stats via gh CLI."""
    stats = {"stars": "?", "forks": "?", "open_issues": "?", "watchers": "?"}
    raw = run_cmd(["gh", "api", f"repos/{REPO}"])
    if raw:
        data = json.loads(raw)
        stats["stars"] = data.get("stargazers_count", "?")
        stats["forks"] = data.get("forks_count", "?")
        stats["open_issues"] = data.get("open_issues_count", "?")
        stats["watchers"] = data.get("subscribers_count", "?")
    return stats


def get_github_traffic() -> dict:
    """Fetch traffic (clones + views) — requires push access."""
    traffic = {"views_14d": "?", "clones_14d": "?", "unique_clones_14d": "?"}
    raw = run_cmd(["gh", "api", f"repos/{REPO}/traffic/views"])
    if raw:
        data = json.loads(raw)
        traffic["views_14d"] = data.get("count", "?")

    raw = run_cmd(["gh", "api", f"repos/{REPO}/traffic/clones"])
    if raw:
        data = json.loads(raw)
        traffic["clones_14d"] = data.get("count", "?")
        traffic["unique_clones_14d"] = data.get("uniques", "?")
    return traffic


def get_pypi_downloads() -> str:
    """Fetch recent download count from PyPI Stats API."""
    try:
        import urllib.request
        url = f"https://pypistats.org/api/packages/{PYPI_PACKAGE}/recent"
        req = urllib.request.Request(url, headers={"User-Agent": "printforge-monitor/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            last_day = data.get("data", {}).get("last_day", "?")
            last_week = data.get("data", {}).get("last_week", "?")
            return f"{last_day} (day) / {last_week} (week)"
    except Exception:
        return "unavailable (not on PyPI yet?)"


def get_recent_issues() -> list[str]:
    """Fetch recent issues via gh CLI."""
    raw = run_cmd(["gh", "issue", "list", "--repo", REPO, "--limit", "5", "--json", "title,createdAt,number"])
    if raw:
        issues = json.loads(raw)
        return [f"  #{i['number']}: {i['title']}" for i in issues]
    return ["  (none or gh CLI not authenticated)"]


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print(f"  PrintForge Launch Monitor — {now}")
    print("=" * 60)

    # GitHub stats
    gh = get_github_stats()
    print(f"\n  GitHub Stars:       {gh['stars']}")
    print(f"  GitHub Forks:       {gh['forks']}")
    print(f"  Open Issues:        {gh['open_issues']}")
    print(f"  Watchers:           {gh['watchers']}")

    # Traffic
    traffic = get_github_traffic()
    print(f"\n  Views (14d):        {traffic['views_14d']}")
    print(f"  Clones (14d):       {traffic['clones_14d']}")
    print(f"  Unique Clones (14d):{traffic['unique_clones_14d']}")

    # PyPI
    pypi = get_pypi_downloads()
    print(f"\n  PyPI Downloads:     {pypi}")

    # Recent issues
    print("\n  Recent Issues:")
    for line in get_recent_issues():
        print(line)

    # Targets comparison
    stars = gh["stars"] if isinstance(gh["stars"], int) else 0
    issues = gh["open_issues"] if isinstance(gh["open_issues"], int) else 0

    print("\n" + "-" * 60)
    print("  Day 1 Targets:")
    print(f"    Stars:  {stars}/100  {'PASS' if stars >= 100 else 'IN PROGRESS' if stars >= 50 else 'EARLY'}")
    print(f"    Issues: {issues}/20   {'PASS' if issues >= 20 else 'IN PROGRESS' if issues >= 5 else 'EARLY'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
