#!/usr/bin/env python
"""Check that the verifier commands of the reproduction queues are the ones that produced the stored results.

For every scripts/verify_ref.py line in the given queue files, the command line is parsed with the parser of
verify_ref.py (nothing is run), its output directory is derived exactly as verify_ref.py does, and the parsed
arguments are compared with the arguments stored in <output dir>/scores.json. Arguments added to verify_ref.py
after a run (absent from its scores.json) are listed separately; their defaults reproduce the old behaviour.

FDU_RUNS=<runs> python reproduce/check_verifier_args.py reproduce/queues/06_verifier.q [...]
"""
import argparse
import json
import os
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import verify_ref  # noqa: E402


class _Stop(Exception):
    pass


def parse(cmd):
    """Return (namespace, output dir) of one verify_ref.py command line without running it."""
    got = {}
    real_parse, real_makedirs = argparse.ArgumentParser.parse_args, os.makedirs

    def parse_args(self, args=None, namespace=None):
        got["ns"] = real_parse(self, args, namespace)
        return got["ns"]

    def makedirs(path, *a, **k):
        raise _Stop(path)

    argparse.ArgumentParser.parse_args, os.makedirs = parse_args, makedirs
    sys.argv = shlex.split(cmd)
    try:
        verify_ref.main()
    except _Stop as s:
        return vars(got["ns"]), s.args[0]
    finally:
        argparse.ArgumentParser.parse_args, os.makedirs = real_parse, real_makedirs
    raise RuntimeError("verify_ref.main() returned before creating its output directory")


def main():
    n_ok = n_missing = n_diff = 0
    for q in sys.argv[1:]:
        for line in open(q):
            line = line.strip()
            if not line.startswith("scripts/verify_ref.py"):
                continue
            ns, out = parse(line)
            rel = os.path.relpath(out, os.environ.get("FDU_RUNS", ROOT))
            sp = os.path.join(out, "scores.json")
            if not os.path.exists(sp):
                n_missing += 1
                print(f"[not on this machine] {rel}")
                continue
            stored = json.load(open(sp))["args"]
            diff = {k: (stored[k], ns.get(k)) for k in stored if str(stored[k]) != str(ns.get(k))
                    and k not in ("num_workers", "gpu", "score_only")}
            new = sorted(k for k in ns if k not in stored)
            if diff:
                n_diff += 1
                print(f"[DIFF] {rel}: stored vs command {diff}")
            else:
                n_ok += 1
                print(f"[ok] {rel}" + (f"  (arguments added later, defaults: {', '.join(new)})" if new else ""))
    print(f"{n_ok} identical, {n_diff} different, {n_missing} not on this machine")


if __name__ == "__main__":
    main()
