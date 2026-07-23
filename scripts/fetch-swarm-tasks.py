#!/usr/bin/env python3
"""Generate swarm-tasks.json from SWE-bench Verified. Run ONCE, anywhere with
internet (laptop or pod), then commit the JSON. The swarm never fetches.

Pulls the psf/requests (8) and pytest-dev/pytest (19) Verified instances
verbatim via the Hugging Face datasets-server API. Zero non-stdlib deps.

Usage: python3 fetch-swarm-tasks.py [out.json]
Default output: swarm-tasks.json NEXT TO THIS SCRIPT (where agent-swarm.sh
reads it), regardless of cwd.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

DATASET = "princeton-nlp/SWE-bench_Verified"
REPOS = ["psf/requests", "pytest-dev/pytest"]
FIELDS = [
    "instance_id",
    "repo",
    "base_commit",
    "environment_setup_commit",
    "version",
    "problem_statement",
]

# Integrity pin: the exact Verified instance ids for these repos.
EXPECTED = {
    "psf__requests-1142", "psf__requests-1724", "psf__requests-1766",
    "psf__requests-1921", "psf__requests-2317", "psf__requests-2931",
    "psf__requests-5414", "psf__requests-6028",
    "pytest-dev__pytest-5262", "pytest-dev__pytest-5631",
    "pytest-dev__pytest-5787", "pytest-dev__pytest-5809",
    "pytest-dev__pytest-5840", "pytest-dev__pytest-6197",
    "pytest-dev__pytest-6202", "pytest-dev__pytest-7205",
    "pytest-dev__pytest-7236", "pytest-dev__pytest-7324",
    "pytest-dev__pytest-7432", "pytest-dev__pytest-7490",
    "pytest-dev__pytest-7521", "pytest-dev__pytest-7571",
    "pytest-dev__pytest-7982", "pytest-dev__pytest-8399",
    "pytest-dev__pytest-10051", "pytest-dev__pytest-10081",
    "pytest-dev__pytest-10356",
}


def get_json(endpoint, params, tries=4):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"https://datasets-server.huggingface.co/{endpoint}?{q}",
        headers={"User-Agent": "profile-swarm-tasks/1.0"},
    )
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}")
            time.sleep(3 * (attempt + 1))


def fetch_repo_rows(repo):
    rows, offset = [], 0
    while True:
        d = get_json("filter", {
            "dataset": DATASET,
            "config": "default",
            "split": "test",
            "where": f"\"repo\"='{repo}'",
            "columns": ",".join(FIELDS),
            "offset": offset,
            "limit": 100,
        })
        rows += [row["row"] for row in d["rows"]]
        if len(rows) >= d["num_rows_total"] or not d["rows"]:
            return rows
        offset = len(rows)


def fetch_all_rows():
    """Fallback: page through the whole split (500 rows), filter locally."""
    rows, offset = [], 0
    while True:
        d = get_json("rows", {
            "dataset": DATASET,
            "config": "default",
            "split": "test",
            "columns": ",".join(FIELDS),
            "offset": offset,
            "length": 100,
        })
        rows += [row["row"] for row in d["rows"]]
        if len(rows) >= d["num_rows_total"] or not d["rows"]:
            return [r for r in rows if r["repo"] in REPOS]
        offset = len(rows)


def main():
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "swarm-tasks.json")
    out_path = sys.argv[1] if len(sys.argv) > 1 else default_out
    try:
        rows = []
        for repo in REPOS:
            got = fetch_repo_rows(repo)
            print(f"{repo}: {len(got)} instances")
            rows += got
    except Exception as e:
        print(f"filter endpoint failed ({e}); falling back to full-split scan")
        rows = fetch_all_rows()
        print(f"full scan: {len(rows)} matching instances")

    got_ids = {r["instance_id"] for r in rows}
    if got_ids != EXPECTED:
        missing = sorted(EXPECTED - got_ids)
        extra = sorted(got_ids - EXPECTED)
        sys.exit(f"FAIL integrity check.\n missing: {missing}\n extra: {extra}")

    for r in rows:
        if not all(r.get(f) for f in FIELDS):
            sys.exit(f"FAIL empty field in {r['instance_id']}")

    rows.sort(key=lambda r: r["instance_id"])
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"OK: {len(rows)} instances -> {out_path}")


if __name__ == "__main__":
    main()
