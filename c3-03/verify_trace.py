"""Compare a C3-03 JSONL trace with an F0 or private M-hist reference (Python 3)."""

import argparse
import json
import math
import sys
from pathlib import Path

WIDTHS = {"h1": 24, "h2": 16, "h3": 8, "logits": 2}
TOL = 2e-6


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"expected a finite number, got {value!r}")
    return value


def read_trace(path):
    rows = {}
    with path.open(encoding="utf-8-sig") as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line.startswith("{"):
                continue  # Serial boot messages and ordinary status lines.
            row = json.loads(line)
            key = row.get("id")
            if not isinstance(key, str) or key in rows:
                raise ValueError(f"line {lineno}: missing, invalid or duplicate id {key!r}")
            rows[key] = row
    return rows


def check_vector(ref, row, worst):
    vid = ref["id"]
    if any(not math.isfinite(x) for x in ref["x"]):
        if row.get("status") != "reject" or any(k in row for k in (*WIDTHS, "d", "decision")):
            raise ValueError(f"{vid}: invalid input must be rejected without inference outputs")
        return

    if row.get("status") != "ok":
        raise ValueError(f"{vid}: finite input must return status ok")
    for stage, width in WIDTHS.items():
        actual, golden = row[stage], ref["lut_reference"][stage]
        if not isinstance(actual, list) or len(actual) != width or len(golden) != width:
            raise ValueError(f"{vid}/{stage}: expected {width} outputs")
        for index, (value, expected) in enumerate(zip(actual, golden)):
            error = abs(number(value) - number(expected)) / max(1.0, abs(expected))
            worst[stage] = max(worst[stage], error)
            if error > TOL:
                raise ValueError(f"{vid}/{stage}[{index}]: error {error:.9g} exceeds {TOL:g}")

    z0, z1 = row["logits"]
    d = number(row["d"])
    # d is printed separately. Allow rounding of float32 and of its decimal encoding.
    if abs(d - (z1 - z0)) > 2.5e-7 * max(1.0, abs(z0), abs(z1), abs(d)):
        raise ValueError(f"{vid}: logged margin is inconsistent with logits")
    label = "person" if z1 > z0 else "no_person"
    golden = ref.get("decision_lut_reference", ref.get("decision_argmax"))
    if row.get("decision") != label or row["decision"] != golden or (d > 0) != (z1 > z0):
        raise ValueError(f"{vid}: inconsistent decision or disagreement with LUT reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    refs = json.loads(args.reference.read_text(encoding="utf-8"))["vectors"]
    if not refs or len({r["id"] for r in refs}) != len(refs):
        raise ValueError("reference has no vectors or duplicate ids")
    rows = read_trace(args.trace)
    ids = {r["id"] for r in refs}
    if rows.keys() != ids:
        raise ValueError(f"missing ids {sorted(ids - rows.keys())}, unexpected ids {sorted(rows.keys() - ids)}")
    worst = dict.fromkeys(WIDTHS, 0.0)
    for ref in refs:
        check_vector(ref, rows[ref["id"]], worst)
    rejected = sum(row.get("status") == "reject" for row in rows.values())
    print(f"PASS {len(refs)}/{len(refs)} ({rejected} rejected inputs)")
    for stage, value in worst.items():
        print(f"{stage}: max normalized error {value:.9g}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
