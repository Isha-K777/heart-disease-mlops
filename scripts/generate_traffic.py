"""Send randomised requests so Grafana dashboards have data.

Usage: python scripts/generate_traffic.py --url http://localhost:8000 --n 200
Includes ~10% invalid payloads to exercise the error panels.
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

BASE = json.load(open("sample_input.json"))


def make_payload():
    p = dict(BASE)
    p.update(age=random.randint(30, 77), sex=random.randint(0, 1), cp=random.randint(1, 4),
             trestbps=random.randint(95, 200), chol=random.randint(125, 400),
             fbs=random.randint(0, 1), restecg=random.randint(0, 2),
             thalach=random.randint(95, 200), exang=random.randint(0, 1),
             oldpeak=round(random.uniform(0, 5), 1), slope=random.randint(1, 3),
             ca=random.randint(0, 3), thal=random.choice([3, 6, 7]))
    if random.random() < 0.1:
        p["age"] = 999  # invalid -> HTTP 422
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--delay", type=float, default=0.05)
    a = ap.parse_args()
    ok = bad = 0
    for _ in range(a.n):
        req = urllib.request.Request(a.url + "/predict", json.dumps(make_payload()).encode(),
                                     {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=5).read()
            ok += 1
        except urllib.error.HTTPError:
            bad += 1
        time.sleep(a.delay)
    print(f"sent {a.n} requests: {ok} ok, {bad} rejected")


if __name__ == "__main__":
    main()
