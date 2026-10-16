#!/usr/bin/env python3
"""Test the fine-tuned IT-support routing model running in Ollama.

Usage:
  python3 test_routing.py                         # score the model on test_set.json (100 held-out tickets)
  python3 test_routing.py --ask "VPN won't connect from home"
  python3 test_routing.py --model it-support --file test_set.json
  python3 test_routing.py --model hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M   # compare with the untuned base

Only the Python standard library is used; Ollama must be running (`ollama serve` or the menu-bar app).
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
EMP_RE = re.compile(r"Assigned to ([^:]+):")
TEAM_RE = re.compile(r"Team:\s*([A-Za-z]+)")


def ask(model, issue, system=None, timeout=180):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": issue})
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 128},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)["message"]["content"].strip()
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach Ollama at {OLLAMA_URL}: {e}\nIs Ollama running? Try: ollama serve")


def parse(text):
    emp = EMP_RE.search(text)
    team = TEAM_RE.search(text)
    return (team.group(1).strip() if team else "?", emp.group(1).strip() if emp else "?")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="it-support", help="Ollama model name (default: it-support)")
    ap.add_argument("--file", default="test_set.json", help="held-out test file (default: test_set.json)")
    ap.add_argument("--ask", help="route a single issue and print the answer")
    ap.add_argument("--system", help="override system prompt (only needed when testing a non-fine-tuned model)")
    ap.add_argument("--limit", type=int, default=0, help="only evaluate the first N test rows")
    args = ap.parse_args()

    if args.ask:
        print(ask(args.model, args.ask, args.system))
        return

    rows = json.load(open(args.file))
    if args.limit:
        rows = rows[: args.limit]

    per_emp = {}
    team_ok = emp_ok = 0
    mistakes = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        want_team, want_emp = parse(r["output"])
        pred = ask(args.model, r["input"], args.system)
        got_team, got_emp = parse(pred)
        per_emp.setdefault(want_emp, [0, 0])
        per_emp[want_emp][1] += 1
        if got_team == want_team:
            team_ok += 1
        if got_emp == want_emp:
            emp_ok += 1
            per_emp[want_emp][0] += 1
        else:
            mistakes.append((r["input"], want_emp, got_emp, pred))
        print(f"\r[{i}/{len(rows)}] employee acc so far: {emp_ok}/{i}", end="", flush=True)
    print()

    n = len(rows)
    print(f"\nModel: {args.model}   ({time.time() - t0:.0f}s)")
    print(f"Team accuracy:     {team_ok}/{n} = {100 * team_ok / n:.1f}%")
    print(f"Employee accuracy: {emp_ok}/{n} = {100 * emp_ok / n:.1f}%")
    print("\nPer employee:")
    for emp, (c, tot) in sorted(per_emp.items()):
        print(f"  {emp:16s} {c:3d}/{tot}")
    if mistakes:
        print("\nMistakes:")
        for q, want, got, pred in mistakes:
            print(f"- {q}\n    expected: {want}\n    got:      {got}\n    raw:      {pred[:160]}")


if __name__ == "__main__":
    main()
