"""Download the public datasets used by the real-data studies.

All files are pinned by SHA-256 so results are reproducible. Datasets are not
committed to git (Criteo alone is 311 MB compressed).

* Criteo Uplift Prediction Dataset v2.1 (Diemert, Betlei, Renaudin, Amini 2018,
  "A Large Scale Benchmark for Uplift Modeling"). 13.98M users from a randomized
  incrementality test: ``treatment`` (assigned to be eligible for ads),
  ``exposure`` (actually shown an ad), ``visit``, ``conversion`` and 12
  anonymized features. License: CC BY-NC-SA 4.0. Mirror of the file published
  by Criteo AI Lab.
* "Marketing A/B testing" ads-vs-PSA experiment (588K users; ``test_group`` ad or
  psa, ``converted``, ``total_ads``, ``most_ads_day``, ``most_ads_hour``). A
  public teaching dataset whose original source is not documented; we treat it
  as a randomized experiment and test that assumption where possible.
* Google Meridian demo data (``national_all_channels.csv``, ``geo_all_channels.csv``):
  the simulated MMM dataset shipped with Google's open-source Meridian MMM.
  Apache-2.0.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"

DATASETS = {
    "criteo-uplift-v2.1.csv.gz": (
        "https://github.com/jszymon/uplift_sklearn_data/releases/download/Criteo/criteo-research-uplift-v2.1.csv.gz",
        "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc",
    ),
    "marketing_ab.csv.gz": (
        "https://github.com/jszymon/uplift_sklearn_data/releases/download/marketing_AB/marketing_AB.csv.gz",
        "a318767a73785b7e54fbf36cbf3004115f61e943844b7ac8d3753a893e093d9d",
    ),
    "meridian_national_all_channels.csv": (
        "https://raw.githubusercontent.com/google/meridian/main/meridian/data/simulated_data/csv/national_all_channels.csv",
        "d001b91e494a90d9c88923df6cbd15fd73af132c3609c424daaea3c57c69803a",
    ),
    "meridian_geo_all_channels.csv": (
        "https://raw.githubusercontent.com/google/meridian/main/meridian/data/simulated_data/csv/geo_all_channels.csv",
        "d9ee016f7cd21f5c90b50da10a794af91a372b69f881d3caa266edd41fdafbf6",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path(name: str) -> Path:
    return RAW / name


def available(name: str) -> bool:
    return path(name).exists()


def fetch(names: list[str] | None = None, verify: bool = True) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in names or list(DATASETS):
        url, digest = DATASETS[name]
        p = path(name)
        if not p.exists():
            print(f"downloading {name} ...", file=sys.stderr)
            urllib.request.urlretrieve(url, p)
        if verify:
            got = sha256(p)
            if got != digest:
                print(f"WARNING {name}: sha256 {got} != pinned {digest} (upstream file changed?)",
                      file=sys.stderr)
            else:
                print(f"ok {name}", file=sys.stderr)
